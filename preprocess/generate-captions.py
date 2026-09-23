import asyncio
import io
import re
import pandas as pd
from ollama import AsyncClient

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_PARQUET = rf"E:\Data\query-earth\git-10mil\git_10m_100.parquet"
OUTPUT_PARQUET = rf"E:\Data\query-earth\git-10mil\captions.parquet"
OLLAMA_MODEL = "ministral-3:3b"

# Adjust concurrency based on GPU VRAM and CPU capacity
MAX_CONCURRENT_IMAGES = 20  # Images processed in parallel


# ==========================================
# PROMPTS & CLEANING HELPERS
# ==========================================
SCENE_PROMPT = "Provide a single-line scene description of this image. Keep it concise, focused, and accurate."

SPATIAL_PROMPT = (
    "Provide a few detailed line (3-5) describing everything in this image. Dont write a prose or story just what you see, and how are they spaially aligned."
    "Emphasize spatial relationships and reasoning (e.g., 'above the big tree is a red house', 'to the left of the road')."
)

KEYWORD_PROMPT = (
    "List all objects, entities, and visible concepts in this image as a comma-separated list. Do not list irrelevant items, like 'earth surface;, mention all objects, no made up stuff."
    "Each keyword or phrase should be 1 to 5 words long. Do not use brackets, bullet points, numbering, or introductory text. "
    "Return ONLY the comma-separated list."
)


def clean_and_deduplicate_keywords(raw_text: str) -> str:
    """Cleans keyword string and deduplicates concepts case-insensitively while preserving original order."""
    if not raw_text:
        return ""

    # Split by comma
    raw_items = raw_text.split(",")
    cleaned_items = []
    seen = set()

    for item in raw_items:
        # Strip whitespace, quotes, and brackets
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
            # Query all 3 captions concurrently for the SAME image
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

            # Clean and deduplicate keywords
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
# MAIN EXECUTION FLOW
# ==========================================
async def main():
    print(f"📖 Reading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)

    if "img_name" not in df.columns or "image_bytes" not in df.columns:
        raise ValueError(
            "Parquet must contain 'id' and 'image_bytes' columns."
        )

    print(
        f"🚀 Starting caption generation for {len(df)} images using '{OLLAMA_MODEL}'..."
    )

    client = AsyncClient()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_IMAGES)

    tasks = [
        process_image_row(client, semaphore, row["img_name"], row["image_bytes"])
        for _, row in df.iterrows()
    ]

    # Gather all results with progress logging
    results = []
    for count, completed_task in enumerate(asyncio.as_completed(tasks), 1):
        res = await completed_task
        results.append(res)
        if count % 10 == 0 or count == len(df):
            print(f" Progress: [{count}/{len(df)}] images completed.")

    # Create output DataFrame
    out_df = pd.DataFrame(results)

    # Filter out failed runs if any, and drop status column
    success_df = out_df[out_df["status"] == "success"].drop(
        columns=["status"]
    )

    print(f"💾 Saving {len(success_df)} rows to {OUTPUT_PARQUET}...")
    success_df.to_parquet(OUTPUT_PARQUET, index=False)
    print("✨ Task complete!")


if __name__ == "__main__":
    asyncio.run(main())