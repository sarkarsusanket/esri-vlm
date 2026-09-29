import io
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_parquet(r"/data/susanket/vlm/images.parquet")


def compute_image_entropy(img_bytes):
    """Calculates Shannon entropy for a JPEG/PNG byte string (0 to 8 bits)."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        histogram = img.histogram()
        total_pixels = sum(histogram)

        entropy = 0.0
        for count in histogram:
            if count > 0:
                p = count / total_pixels
                entropy -= p * math.log2(p)
        return entropy
    except Exception:
        return None


def plot_sample_entropies_grid(
    df, sample_size=100, cols=10, sort_by_entropy=True, save_path="image.png"
):
    # 1. Sample random valid rows
    sampled_df = df.sample(n=min(sample_size, len(df))).copy()

    # 2. Compute entropy for all sampled images
    print("Computing image entropies...")
    sampled_df["entropy"] = sampled_df["image_bytes"].apply(
        compute_image_entropy
    )

    # 3. Sort by entropy (lowest to highest)
    if sort_by_entropy:
        sampled_df = sampled_df.sort_values(
            by="entropy", ascending=True
        ).reset_index(drop=True)

    # 4. Calculate grid dimensions
    total_imgs = len(sampled_df)
    rows = math.ceil(total_imgs / cols)

    # 5. Create figure grid
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.2, rows * 2.5), dpi=150
    )
    axes = axes.flatten() if total_imgs > 1 else [axes]

    for idx, row in sampled_df.iterrows():
        ax = axes[idx]
        byte_data = row["image_bytes"]
        entropy = row["entropy"]

        if byte_data and entropy is not None:
            try:
                img = Image.open(io.BytesIO(byte_data))
                width, height = img.size  # Extract resolution in px

                ax.imshow(img)

                # Determine badge background color based on entropy score
                if entropy < 4.0:
                    bg_color = "#ff4d4d"  # Red (Low entropy / low variance)
                elif entropy < 6.0:
                    bg_color = "#ffa64d"  # Amber
                else:
                    bg_color = "#2eb82e"  # Green

                # Overlay pill with Resolution and Entropy
                label_text = f"E: {entropy:.2f}\n{width}×{height}px"

                ax.text(
                    0.5,
                    0.05,
                    label_text,
                    transform=ax.transAxes,
                    fontsize=7,
                    fontweight="bold",
                    color="white",
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=bg_color,
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )
            except Exception:
                ax.text(
                    0.5,
                    0.5,
                    "Corrupt",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="red",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )

        ax.axis("off")

    # Hide unused subplot slots
    for i in range(total_imgs, len(axes)):
        axes[i].axis("off")

    # Layout adjustments
    plt.suptitle(
        f"Image Entropy & Resolution Overview (Sorted Low → High) | N={total_imgs}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"Saved visualization grid to {save_path}")
    plt.show()


# Run visualization for 100 images in a 10x10 grid
plot_sample_entropies_grid(
    df, sample_size=100, cols=10, sort_by_entropy=True, save_path="image.png"
)