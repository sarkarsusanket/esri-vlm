import io
import math
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import pandas as pd

# Load dataset
df = pd.read_parquet(r"/data/susanket/vlm/images.parquet")


def compute_image_entropy(img_bytes):
    """Calculates Shannon entropy for a JPEG/PNG byte string (0 to 8 bits)."""
    if not img_bytes or not isinstance(img_bytes, bytes):
        return None
    try:
        # Load byte stream into PIL and convert to grayscale
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        histogram = img.histogram()
        total_pixels = sum(histogram)

        if total_pixels == 0:
            return 0.0

        entropy = 0.0
        for count in histogram:
            if count > 0:
                p = count / total_pixels
                entropy -= p * math.log2(p)
        return entropy
    except Exception:
        return None


def add_entropy_column(
    dataframe, image_col="image_bytes", num_workers=None, chunksize=100
):
    """Calculates entropy in parallel across all CPU cores and appends an 'entropy' column."""
    print(f"Calculating entropy for {len(dataframe)} images...")

    image_bytes_list = dataframe[image_col].tolist()

    # Process byte lists across available CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        entropies = list(
            executor.map(compute_image_entropy, image_bytes_list, chunksize=chunksize)
        )

    # Assign new column
    dataframe["entropy"] = entropies
    print("Done!")
    return dataframe


# Run parallel calculation
df = add_entropy_column(df, image_col="image_bytes")

# Check results
print(df[["image_bytes", "entropy"]].head())

# Filter out low entropy images (e.g., threshold < 4.5)
# clean_df = df[df["entropy"] >= 4.5].reset_index(drop=True)

df.to_parquet(rf"/data/susanket/vlm/images_with_entropy.parquet")