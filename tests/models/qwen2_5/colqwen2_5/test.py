import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

model_name = "vidore/colqwen2.5-v0.2"

model = ColQwen2_5.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(model_name)


# Your inputs
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "What is the organizational structure for our R&D department?",
    "Can you provide a breakdown of last year’s financial performance?",
]

batch_images = processor.process_documents(images).to(model.device)
batch_queries = processor.process_text(queries).to(model.device)

print("batch_images shape:", batch_images.shape if hasattr(batch_images, "shape") else type(batch_images))
print("batch_queries shape:", batch_queries.shape if hasattr(batch_queries, "shape") else type(batch_queries))

def print_shape(name, tensor):
    if isinstance(tensor, torch.Tensor):
        print(f"{name} shape: {tensor.shape}")
    else:
        print(f"{name} is type {type(tensor)}")

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

# Print shapes
print_shape("image_embeddings", image_embeddings)
print_shape("query_embeddings", query_embeddings)

# Scores
scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print_shape("scores", scores)

print(scores)