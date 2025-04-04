from typing import ClassVar, List, Optional, Tuple, Union

import torch
import math
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class ColQwen2_5_Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):  # noqa: N801
    """
    Processor for ColQwen2.5.

    Args:
        *args: Variable length argument list to be passed to the parent `Qwen2VLProcessor` class.
        max_num_visual_tokens: The maximum number of visual tokens that can be processed by the model.
        **kwargs: Arbitrary keyword arguments to be passed to the parent `Qwen2VLProcessor` class.
    """

    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def __init__(
        self,
        *args,
        max_num_visual_tokens: int = 768,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

        self.max_num_visual_tokens = max_num_visual_tokens
        self.factor = 28
        self.min_pixels = 4 * 28 * 28
        self.max_pixels = self.max_num_visual_tokens * 28 * 28

        self.image_processor.min_pixels = self.min_pixels
        self.image_processor.max_pixels = self.max_pixels

    def process_images(self, images: List[Image.Image]) -> BatchFeature:
        """
        Process images for ColQwen2.5.
        """
        texts_doc = [self.visual_prompt_prefix] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = self(
            text=texts_doc,
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)

        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)

        return batch_doc

    def process_queries(
        self,
        text: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        """
        Process queries for ColQwen2.5.

        NOTE: `max_length` is not used and kept only for trainer compatibility.
        """
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for query in text:
            query = self.query_prefix + query + suffix
            texts_query.append(query)

        batch_query = self(
            text=texts_query,
            return_tensors="pt",
            padding="longest",
        )

        return batch_query
    
    def extract_images_from_document(self, document: Image.Image) -> List[Image.Image]:
        """
        Extracts images from a document page using OpenCV.
        This function detects and extracts potential image regions inside the document.
        """
        extracted_images = []

        cv_img = np.array(document)

        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding to separate text and objects
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find the biggest non-text object (the image inside)
        image_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            # Skip small regions and extreme aspect ratios (text boxes)
            if w * h > 5000 and 0.5 < aspect_ratio < 2:
                image_contours.append((x, y, w, h))

        # Display all valid cropped images
        if image_contours:
            for (x, y, w, h) in image_contours:
                extracted_img = cv_img[y:y+h, x:x+w]
                pil_img = Image.fromarray(extracted_img)
                extracted_images.append(pil_img)
        else:
            print("No image detected!")

        return extracted_images
    
    def image_to_patch_tensor(self, image):
        """
        Converts a PIL image into a tensor formatted as a patch of size (1176,).
        The image is dynamically resized to (a, b) where a * b = 392.
        """
        # Compute best (a, b) pair such that a * b = 392
        a = int(math.sqrt(392))  # Start with sqrt(392) as a reasonable initial guess
        while 392 % a != 0:  # Find the closest factor pair
            a -= 1
        b = 392 // a  # Compute corresponding b

        # Define transformation
        transform = transforms.Compose([
            transforms.Resize((a, b)),  # Resize to (a, b)
            transforms.ToTensor(),  # Convert to (C, H, W) tensor
            transforms.Normalize((0.5,), (0.5,)),  # Normalize
        ])

        # Apply transformation
        tensor_image = transform(image)  # Shape: (3, a, b)

        # Flatten to shape (3 * a * b = 1176,)
        tensor_patch = tensor_image.view(-1)

        return tensor_patch  # Shape: (1176,)

    
    def process_documents(self, document: Image.Image) -> List[BatchFeature]:
        """
        Process documents by extracting text and images, converting them into tensor patches,
        and ensuring they are properly batched.
        """
        images = [document]
        extracted_images = self.extract_images_from_document(document)
        images.extend(extracted_images)

        images = [image.convert("RGB") for image in images]
        
        batches = []
        for image in images:
            # texts_doc = [self.visual_prompt_prefix]
            batch = self(
                text=self.visual_prompt_prefix,
                images=image,
                padding="longest",
                return_tensors="pt",
            )

            offsets = batch["image_grid_thw"][:, 1] * batch["image_grid_thw"][:, 2]  # (batch_size,)
            pixel_values = list(torch.split(batch["pixel_values"], offsets.tolist()))

            # Pad the list of pixel_value tensors to the same length
            batch["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
                pixel_values, batch_first=True
            )  # (batch_size, max_num_patches, pixel_values)

            batches.append(batch)

        return batches  # Return the processed inputs for further use

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id

    def embed_multimodal_query(
        self,
        text: str,
        images: List[Image.Image],
        mock_image_size: tuple = (224, 224)
    ) -> torch.Tensor:
        """
        Embeds multimodal queries by fusing text and image representations [[9]].
    
        Args:
            text: Text component of the query.
            images: List of images in the query.
            mock_image_size: Size for placeholder image during text processing [[2]].
        
        Returns:
            Combined multimodal embedding tensor.
        """
        # Process text with mock white image for format consistency [[2]]
        mock_image = Image.new('RGB', mock_image_size, color=(255, 255, 255))
        text_inputs = self(
            text=text,
            # images=[mock_image],
            return_tensors="pt",
            padding="longest"
        )
    
        # Process images with visual prompt prefix [[6]]
        texts_doc = [self.visual_prompt_prefix] * len(images)
        images = [image.convert("RGB") for image in images]

        image_inputs = self(
            text=texts_doc,
            images=images,
            padding="longest",
            return_tensors="pt",
        )

        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        offsets = image_inputs["image_grid_thw"][:, 1] * image_inputs["image_grid_thw"][:, 2]  # (batch_size,)

        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(image_inputs["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        image_inputs["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)

    
        # Generate embeddings with VLM
        # with torch.no_grad():
        #     text_outputs = self.model(**text_inputs)
        #     image_outputs = self.model(**image_inputs)
    
        # # Late fusion via concatenation [[1]][[6]]
        # text_emb = text_outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Shape: [embed_dim]
        # image_emb = image_outputs.last_hidden_state.mean(dim=[0, 1])      # Shape: [embed_dim]
    
        # # Concatenate along the feature dimension [[1]][[6]]
        # combined_emb = torch.cat([text_emb, image_emb], dim=0)  # Shape: [2 * embed_dim]
    
        return text_inputs, image_inputs  # Return the processed inputs for further use
    
    def concatenate_embeddings(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """
        Concatenate text and image embeddings along the feature dimension.
    
        Args:
            text_emb: Text embedding tensor.
            image_emb: Image embedding tensor.
        
        Returns:
            Concatenated embedding tensor.
        """
        return torch.cat([text_emb, image_emb], dim=0)