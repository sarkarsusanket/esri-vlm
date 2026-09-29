import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import asyncio
import io
import os
import re
import pandas as pd
from ollama import AsyncClient

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_PARQUET = r"/data/susanket/vlm/images/image-02.parquet"
OUTPUT_PARQUET = r"/data/susanket/vlm/captions/caption-02.parquet"
OLLAMA_MODEL = "ministral-3:3b"

# Multi-GPU Endpoints
OLLAMA_ENDPOINTS = [
    "http://127.0.0.1:11434",
    # "http://127.0.0.1:11435",
    # "http://127.0.0.1:11436",
    # "http://127.0.0.1:11437",
]

# Adjust concurrency (total concurrent requests across all 4 GPUs)
MAX_CONCURRENT_IMAGES = 160  # e.g., ~40 active per GPU
BATCH_SIZE = 1000            # Save incrementally every 1000 images


# ==========================================
# PROMPTS & CLEANING HELPERS
# ==========================================
# ==========================================
# REFINED PROMPTS
# ==========================================

SCENE_PROMPT = (
    "You are analyzing an overhead aerial/satellite image. "
    "Provide a single concise sentence summarizing the primary land cover, geographic terrain, or urban settlement type visible. "
    "Be direct and objective. Do not guess specific real-world location names or unverified details."
)

SPATIAL_PROMPT = (
    "You are analyzing an overhead aerial/satellite image. "
    "Write a detailed 3-to-5 sentence spatial description detailing the visible features and how they are positioned relative to one another.\n\n"
    "CRITICAL RULES:\n"
    "1. Focus purely on observable spatial layouts using explicit directions (e.g., top-left, center, running north-south, adjacent to).\n"
    "2. Do NOT write prose, backstory, or speculation.\n"
    "3. Do NOT use numbered lists, bullet points, or markdown headers. Return plain continuous text sentences only.\n"
    "4. If a feature is unclear, describe its visual attributes (e.g., 'a dark linear structure') rather than inventing a specific label.\n\n"
    "EXAMPLE OUTPUT:\n"
    "In the central region, a dense cluster of residential buildings with reddish rooftops is visible. "
    "Running along the eastern edge from top-right to bottom-right is a paved two-lane road lined with sparse vegetation. "
    "To the west of the buildings, a large rectangular plot of agricultural land stretches toward the boundary. "
    "A small patch of dense green foliage occupies the bottom-left corner, directly adjacent to an unpaved dirt path."
)

KEYWORD_PROMPT = (
    "Identify all clearly visible land cover types, structures, surface features, and geographic concepts in this aerial image. "
    "Return ONLY a single, comma-separated list of short keywords or phrases (1 to 4 words each).\n\n"
    "CRITICAL RULES:\n"
    "1. Do NOT include introductory text, conversational filler, markdown formatting, or labels like 'Keywords:'.\n"
    "2. Do NOT guess abstract features or unverified items (e.g., avoid 'earth surface', 'nature', 'location').\n"
    "3. If the image is blurred or unidentifiable, respond with only: 'unclear image'.\n"
    "4. Format strictly as item1, item2, item3."
)


def clean_and_deduplicate_keywords(raw_text: str) -> str:
    """Cleans keyword string and deduplicates concepts case-insensitively while preserving original order."""
    if not raw_text:
        return ""

    raw_items = raw_text.split(",")
    cleaned_items = []
    seen = set()

    for item in raw_items:
        cleaned = re.sub(r"[\[\]\(\)\"']", "", item).strip()

        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            cleaned_items.append(cleaned)

    return ", ".join(cleaned_items)


# ==========================================
# ASYNC WORKERS
# ==========================================
async def generate_single_caption(
    client: AsyncClient, img_bytes: bytes, prompt: str
) -> str:
    """Helper to query Ollama asynchronously for a single prompt."""
    response = await client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [img_bytes],
            }
        ],
    )
    return response.message.content.strip()


async def process_image_row(
    client: AsyncClient,
    semaphore: asyncio.Semaphore,
    img_id: str | int,
    img_bytes: bytes,
) -> dict:
    """Processes one image by requesting scene, spatial, and keyword captions concurrently."""
    async with semaphore:
        try:
            scene_coro = generate_single_caption(client, img_bytes, SCENE_PROMPT)
            spatial_coro = generate_single_caption(
                client, img_bytes, SPATIAL_PROMPT
            )
            keyword_coro = generate_single_caption(
                client, img_bytes, KEYWORD_PROMPT
            )

            scene_desc, spatial_desc, raw_keywords = await asyncio.gather(
                scene_coro, spatial_coro, keyword_coro
            )

            clean_keywords = clean_and_deduplicate_keywords(raw_keywords)

            return {
                "id": img_id,
                "scene_caption": scene_desc,
                "spatial_dense_caption": spatial_desc,
                "keywords": clean_keywords,
                "status": "success",
            }
        except Exception as e:
            print(f"⚠️ Error processing ID '{img_id}': {e}")
            return {
                "id": img_id,
                "scene_caption": "",
                "spatial_dense_caption": "",
                "keywords": "",
                "status": f"failed: {str(e)}",
            }


# ==========================================
# INCREMENTAL SAVE HELPER
# ==========================================
def append_to_parquet(df_batch: pd.DataFrame, file_path: str):
    """Appends a pandas DataFrame to an existing Parquet file, or creates it if missing."""
    if not os.path.exists(file_path):
        df_batch.to_parquet(file_path, engine="fastparquet", index=False)
    else:
        df_batch.to_parquet(file_path, engine="fastparquet", append=True, index=False)


# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
async def main():
    print(f"📖 Reading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)

    if "file_name" not in df.columns or "image_bytes" not in df.columns:
        raise ValueError(
            "Parquet must contain 'file_name' and 'image_bytes' columns."
        )

    # Resume capability: Check for already processed image IDs
    processed_ids = set()
    if os.path.exists(OUTPUT_PARQUET):
        try:
            existing_df = pd.read_parquet(OUTPUT_PARQUET, columns=["id"])
            processed_ids = set(existing_df["id"].values)
            print(f"🔄 Resuming execution: Found {len(processed_ids)} already processed images.")
        except Exception as e:
            print(f"⚠️ Could not read existing output parquet: {e}")

    # Filter out already completed items
    df_remaining = df[~df["file_name"].isin(processed_ids)]
    total_remaining = len(df_remaining)

    if total_remaining == 0:
        print("✨ All images have already been processed!")
        return

    print(
        f"🚀 Processing {total_remaining} remaining images across {len(OLLAMA_ENDPOINTS)} GPUs using '{OLLAMA_MODEL}'..."
    )

    # Initialize a pool of AsyncClients (one per GPU endpoint)
    clients = [AsyncClient(host=endpoint) for endpoint in OLLAMA_ENDPOINTS]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGES)

    # Process images in chunks of BATCH_SIZE
    rows = df_remaining.to_dict("records")
    for batch_idx in range(0, total_remaining, BATCH_SIZE):
        batch_rows = rows[batch_idx : batch_idx + BATCH_SIZE]
        
        print(f"\n📦 Processing batch {batch_idx // BATCH_SIZE + 1} ({len(batch_rows)} images)...")

        # Assign round-robin clients across GPUs
        tasks = [
            process_image_row(
                clients[idx % len(clients)], 
                semaphore, 
                row["file_name"], 
                row["image_bytes"]
            )
            for idx, row in enumerate(batch_rows)
        ]

        batch_results = []
        for count, completed_task in enumerate(asyncio.as_completed(tasks), 1):
            res = await completed_task
            batch_results.append(res)
            if count % 50 == 0 or count == len(batch_rows):
                print(f"  Batch Progress: [{count}/{len(batch_rows)}] images done.")

        # Filter out failed rows
        batch_df = pd.DataFrame(batch_results)
        success_df = batch_df[batch_df["status"] == "success"].drop(
            columns=["status"]
        )

        if not success_df.empty:
            print(f"💾 Appending {len(success_df)} rows to {OUTPUT_PARQUET}...")
            append_to_parquet(success_df, OUTPUT_PARQUET)

    print("\n✨ All batches completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())