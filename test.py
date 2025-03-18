import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

from datasets import load_dataset

m2kr_dataset = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", split="test")

model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# # Your inputs
# images = [
#     Image.new("RGB", (128, 128), color="white"),
#     Image.new("RGB", (64, 32), color="black"),
# ]
# queries = [
#     "What is the organizational structure for our R&D department?",
#     "Can you provide a breakdown of last year’s financial performance?",
# ]

# # Process the inputs
# batch_images = processor.process_images(images).to(model.device)
# batch_queries = processor.process_queries(queries).to(model.device)

# # Forward pass
# with torch.no_grad():
#     image_embeddings = model(**batch_images)
#     query_embeddings = model(**batch_queries)

# scores = processor.score_multi_vector(query_embeddings, image_embeddings)

for example in m2kr_dataset:
    text = example["text"]
    images = example["images"]  # List[Image.Image]
    
    # Generate concatenated embedding
    embedding = processor.embed_multimodal_query(
        text=text,
        images=images,
        mock_image_size=(224, 224)
    )  # Shape: (2 * embed_dim,)