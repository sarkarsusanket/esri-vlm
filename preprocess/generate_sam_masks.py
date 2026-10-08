import io
import os
import csv
import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from PIL import Image

import sys
sys.path.append(rf"/home/susanket/esri-vlm/preprocess/sam3")

# SAM 3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Configuration & Paths ---
IMAGES_PARQUET = r"/data/susanket/vlm/images/image-02.parquet"
CAPTIONS_PARQUET = r"/data/susanket/vlm/captions/caption-02.parquet"
OUTPUT_MASK_DIR = r"/data/susanket/vlm/masks/masks-02"
OUTPUT_CSV = r"/data/susanket/vlm/masks/mask-02.csv"
MAX_PROMPT_WORKERS = 50  # Threads for running text prompts concurrently

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Create masks directory if it doesn't exist
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


# --- 1. Global SAM 3 Model Initialization ---
print(f"Loading SAM 3 model on {DEVICE}...")
sam3_model = build_sam3_image_model().to(device=DEVICE)
sam3_processor = Sam3Processor(sam3_model)


# --- 2. Single Keyword Prompt Inference ---
def predict_single_prompt(processor, state, prompt: str):
    """Executes SAM 3 text prompt over precomputed image embeddings."""
    with torch.cuda.amp.autocast(dtype=DTYPE):
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output.get("masks", [])
    if len(masks) == 0:
        return prompt, None

    masks_np = masks.detach().cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
    masks_np = np.squeeze(masks_np)

    # Convert probability/logit mask to boolean mask
    if masks_np.ndim == 2:
        combined_mask = masks_np > 0.5
    elif masks_np.ndim > 2:
        combined_mask = np.any(masks_np > 0.5, axis=0)
    else:
        return prompt, None

    # Check if any pixels were actually segmented
    if not np.any(combined_mask):
        return prompt, None

    return prompt, combined_mask


# --- 3. Main Processing Pipeline ---
def main():
    print("Loading Parquet files...")
    img_df = pd.read_parquet(IMAGES_PARQUET)
    cap_df = pd.read_parquet(CAPTIONS_PARQUET).rename(columns={"id": "file_name"})

    # Merge on file_name
    df = pd.merge(img_df, cap_df, on="file_name")

    print(f"Total merged rows: {len(df)}")

    # Initialize CSV Output Writer
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "keywords"])  # CSV Header

        for idx, row in tqdm.tqdm(df.iterrows()):
            file_name = str(row["file_name"])
            raw_keywords = str(row.get("keywords", ""))

            # Skip condition 1: "unclear image" present in keywords
            if "unclear image" in raw_keywords.lower():
                print(f"[{idx+1}/{len(df)}] Skipping '{file_name}': Image marked as unclear.")
                continue

            # Parse keywords list
            keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
            if not keywords:
                continue

            # Load Image
            img_bytes = row["image_bytes"]
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"Error loading image for {file_name}: {e}")
                continue

            # Compute Heavy Image Embeddings ONCE per image
            with torch.cuda.amp.autocast(dtype=DTYPE):
                inference_state = sam3_processor.set_image(image)

            # Parallel inference for keywords across image feature map
            valid_keywords = []
            valid_masks = []

            with ThreadPoolExecutor(max_workers=MAX_PROMPT_WORKERS) as executor:
                futures = [
                    executor.submit(predict_single_prompt, sam3_processor, inference_state, kw)
                    for kw in keywords
                ]
                results = [f.result() for f in futures]

            # Filter and align valid keywords with non-empty masks
            for kw, mask in results:
                if mask is not None:
                    valid_keywords.append(kw)
                    valid_masks.append(mask)

            # Skip saving if no valid masks were generated
            if not valid_masks:
                print(f"[{idx+1}/{len(df)}] '{file_name}': No masks generated for keywords.")
                continue

            # Stack masks into a 3D numpy array: shape -> (N_masks, Height, Width)
            masks_array = np.stack(valid_masks, axis=0)

            # Save NumPy binary mask file
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            npy_path = os.path.join(OUTPUT_MASK_DIR, f"{base_name}.npy")
            np.save(npy_path, masks_array)

            # Write synced row to CSV
            keywords_str = ", ".join(valid_keywords)
            writer.writerow([file_name, keywords_str])

            print(f"[{idx+1}/{len(df)}] Processed '{file_name}': Saved {len(valid_keywords)} masks.")

    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()