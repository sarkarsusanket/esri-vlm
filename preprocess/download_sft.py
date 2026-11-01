from datasets import load_dataset
import pandas as pd
import os
import io

# Configuration
REPO_ID = "lcybuaa/Git-10M"
BLOCK_SIZE = 1_000_000  # 1 Million rows per block
OUTPUT_DIR = "/data/susanket/vlm/sft/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Initializing streaming for {REPO_ID}...")
dataset = load_dataset(REPO_ID, split="train", streaming=True)

current_block = []
block_index = 1

print(f"Starting download and conversion in blocks of {BLOCK_SIZE:,} rows...")

for i, sample in enumerate(dataset):
    # Convert PIL Image to raw bytes so PyArrow/Parquet can serialize it
    if "image" in sample and sample["image"] is not None:
        img = sample["image"]
        img_byte_arr = io.BytesIO()
        # Save image to bytes buffer (using JPEG format)
        img.save(img_byte_arr, format=img.format if img.format else "JPEG")
        sample["image"] = img_byte_arr.getvalue()
    
    current_block.append(sample)
    
    # Once a block reaches 1M items, save it to disk
    if len(current_block) == BLOCK_SIZE:
        output_file = os.path.join(OUTPUT_DIR, f"block_{block_index:02d}.parquet")
        print(f"Saving block {block_index} ({len(current_block):,} rows) to {output_file}...")
        
        df = pd.DataFrame(current_block)
        df.to_parquet(output_file, index=False)
        
        # Reset for the next block
        current_block = []
        block_index += 1

# Save any remaining samples that didn't fill a full block
if current_block:
    output_file = os.path.join(OUTPUT_DIR, f"block_{block_index:02d}.parquet")
    print(f"Saving final block {block_index} ({len(current_block):,} rows) to {output_file}...")
    
    df = pd.DataFrame(current_block)
    df.to_parquet(output_file, index=False)

print("Download complete! All blocks saved in:", OUTPUT_DIR)