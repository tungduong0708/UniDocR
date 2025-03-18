from datasets import load_dataset
import os
import random

# Path to your image folder
wiki_folder = "./screenshots"
image_query_folder = "./image_queries"

# Load dataset
WIT_ds = load_dataset("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", "WIT_passages")
train_ds = WIT_ds['train_passages']

wiki_folder = [os.path.join(wiki_folder, f) for f in os.listdir(wiki_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
image_query_folder = [os.path.join(image_query_folder, f) for f in os.listdir(image_query_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

# Convert dataset to a dictionary mapping pos_item_ids to img_path and instruction
pos_item_dict = {
    pos_id: {"img_path": img_path, "instruction": instruction}
    for pos_id, img_path, instruction in zip(train_ds['pos_item_ids'], train_ds['img_path'], train_ds['instruction'])
}

# Create list of lists with variant lengths (5 to 200)
image_lists = []
i = 0
while i < len(image_query_folder):
    # Random length between 5 and 200
    group_size = random.randint(5, 200)
    if (i + group_size) > len(image_query_folder):
        group_size = len(image_query_folder) - i
    # Ensure we don't go beyond the list length
    image_lists.append(image_query_folder[i:i + group_size])
    i += group_size