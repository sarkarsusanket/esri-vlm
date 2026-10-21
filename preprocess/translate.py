import os
import random
from pathlib import Path
import polars as pl
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

# Define target languages and their M2M100 language codes
# Note: M2M100 uses standard ISO codes (e.g., 'fr' for French, 'zh' for Simplified Chinese)
LANGUAGES = {
    "French": "fr",
    "Portuguese": "pt",
    "German": "de",
    "Spanish": "es",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh",  # Handled via m2m100 chinese
    "Cantonese": "yue",
}

# Load Model & Tokenizer
MODEL_NAME = "facebook/m2m100_418M"  # Use "facebook/m2m100_1.2B" for higher quality
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model onto {DEVICE}...")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)


def translate_batch(texts: list[str], target_lang_code: str) -> list[str]:
    """Translates a batch of texts into a single target language using M2M100."""
    tokenizer.src_lang = "en"  # Assuming source caption column is English
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(DEVICE)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(target_lang_code),
        max_length=128,
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def process_parquet_folder(
    input_folder: str, output_folder: str, caption_col: str = "caption", batch_size: int = 1024
):
    """Reads all parquet files in input_folder, randomly assigns one target language

    per row/batch, translates the caption, and writes to output_folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_path.glob("*.parquet"))
    lang_names = list(LANGUAGES.keys())

    print(f"Found {len(parquet_files)} parquet file(s) to process.")

    for file in tqdm(parquet_files, desc="Processing Parquet files"):
        df = pl.read_parquet(file)

        if caption_col not in df.columns:
            print(f"Skipping {file.name}: Column '{caption_col}' not found.")
            continue

        num_rows = len(df)

        # Randomly sample one target language name for each row
        assigned_languages = [random.choice(lang_names) for _ in range(num_rows)]

        # Group indices by assigned language to run fast batched inference per language
        lang_groups = {}
        for idx, lang in enumerate(assigned_languages):
            lang_groups.setdefault(lang, []).append(idx)

        captions = df[caption_col].to_list()
        translated_results = [""] * num_rows

        # Perform translation batch-by-batch for each language group
        for lang_name, indices in tqdm(lang_groups.items()):
            lang_code = LANGUAGES[lang_name]
            group_captions = [captions[i] for i in indices]

            # Mini-batch processing to prevent GPU OOM
            translated_group = []
            for i in tqdm(range(0, len(group_captions), batch_size)):
                sub_batch = group_captions[i : i + batch_size]
                translated_sub = translate_batch(sub_batch, lang_code)
                translated_group.extend(translated_sub)

            # Map back translated strings to their original row index
            for idx, trans_text in zip(indices, translated_group):
                translated_results[idx] = trans_text

        # Add new columns using Polars
        updated_df = df.with_columns(
            [
                pl.Series("translated_col", translated_results),
                pl.Series("translated_lang", assigned_languages),
            ]
        )

        # Save to output path
        out_file = output_path / f"translated_{file.name}"
        updated_df.write_parquet(out_file)

    print("All files processed successfully!")


if __name__ == "__main__":
    # Specify your paths here
    INPUT_FOLDER = "/data/susanket/vlm/sft/"
    OUTPUT_FOLDER = "/data/susanket/vlm/sft/"
    CAPTION_COLUMN = "caption"  # Change to your caption column name

    process_parquet_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        caption_col=CAPTION_COLUMN,
        batch_size=512,  # Adjust batch_size depending on your VRAM
    )