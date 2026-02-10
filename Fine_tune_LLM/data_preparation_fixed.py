#!/usr/bin/env python3
"""
Prepare Rick Sanchez (Rick and Morty) dialogue data for fine-tuning.
Downloads dataset from Hugging Face and filters for Rick's lines.
"""

import os
import yaml
import time
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Hugging Face Dataset ID
HF_DATASET_ID = "Prarabdha/Rick_and_Morty_Transcript"


def load_config():
    """Load configuration from `config.yaml` or fallback to `config_fixed.yaml`.

    Returns:
        dict: Parsed YAML configuration.

    Raises:
        FileNotFoundError: If neither config file exists.
    """
    # Check current working directory first, then the script directory
    script_dir = Path(__file__).resolve().parent
    candidates = [Path(p) for p in ("config.yaml", "config_fixed.yaml")]
    for cfg in candidates:
        # 1) cwd
        if cfg.exists():
            with open(cfg, "r") as f:
                return yaml.safe_load(f)
        # 2) alongside script
        alt = script_dir / cfg
        if alt.exists():
            with open(alt, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "No configuration file found. Create 'config.yaml' or 'config_fixed.yaml' in the project root or the script folder."
    )


def is_valid_dialogue(text: str, min_length: int = 10) -> bool:
    """
    Filter out low-quality dialogue.

    Args:
        text: Dialogue text to validate
        min_length: Minimum character length

    Returns:
        True if dialogue is valid, False otherwise
    """
    text = text.strip()

    # Too short
    if len(text) < min_length:
        return False

    # Stage directions or actions in brackets
    if text.startswith("[") or text.endswith("]"):
        return False

    # Likely action descriptions (e.g., *burps*, *stutters*)
    if text.count("*") > 2:
        return False

    return True


def load_and_process_data(max_retries: int = 3):
    """
    Load Rick and Morty dataset and extract Rick's dialogue.

    Args:
        max_retries: Maximum number of retry attempts

    Returns:
        List of Rick's dialogue lines
    """
    for attempt in range(max_retries):
        try:
            print(f"Loading dataset {HF_DATASET_ID} from Hugging Face...")
            print(f"Attempt {attempt + 1} of {max_retries}")

            # Load dataset
            dataset = load_dataset(HF_DATASET_ID)

            # Convert to pandas for easier filtering
            df = dataset["train"].to_pandas()

            print(f"✓ Loaded {len(df)} total lines.")
            print(f"Columns: {df.columns.tolist()}")

            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]

            # Map dataset columns to standard names
            # Dataset uses 'speaker' and 'dialouge' (sic)
            if "speaker" in df.columns:
                df = df.rename(columns={"speaker": "character"})
            if "dialouge" in df.columns:
                df = df.rename(columns={"dialouge": "line"})

            # Validate required columns exist
            if "character" not in df.columns or "line" not in df.columns:
                raise ValueError(
                    f"Expected 'character' and 'line' columns. "
                    f"Found: {df.columns.tolist()}"
                )

            # Filter for Rick's lines
            rick_lines = df[df["character"] == "Rick"]

            print(f"✓ Found {len(rick_lines)} lines for Rick Sanchez.")

            # Fallback: try loose matching if no exact matches
            if len(rick_lines) == 0:
                print(
                    "⚠️  Warning: No exact matches for 'Rick'. Trying loose matching..."
                )
                print(f"Available characters: {df['character'].unique()[:10]}")
                rick_lines = df[
                    df["character"].str.contains("Rick", case=False, na=False)
                ]
                print(f"✓ Found {len(rick_lines)} lines with loose matching.")

            if len(rick_lines) < 50:
                raise ValueError(
                    f"Insufficient data: only {len(rick_lines)} lines found. "
                    f"Need at least 50 for meaningful fine-tuning."
                )

            return rick_lines["line"].tolist()

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("❌ All retry attempts exhausted.")
                raise


def format_for_training(dialogues, tokenizer, config):
    """
    Format dialogue for training.

    Args:
        dialogues: List of dialogue strings
        tokenizer: Tokenizer instance
        config: Configuration dict

    Returns:
        List of formatted training examples
    """
    formatted_data = []
    min_length = config["data"].get("min_dialogue_length", 10)

    for line in dialogues:
        text = str(line).strip()

        # Apply quality filters
        if not is_valid_dialogue(text, min_length):
            continue

        # Format: Rick Sanchez: {Line}<|endoftext|>
        formatted_text = f"Rick Sanchez: {text}{tokenizer.eos_token}"
        formatted_data.append({"text": formatted_text})

    print(f"✓ Filtered to {len(formatted_data)} valid training examples.")
    return formatted_data


def create_tokenized_dataset(formatted_data, tokenizer, config):
    """
    Create and tokenize dataset.

    Args:
        formatted_data: List of formatted examples
        tokenizer: Tokenizer instance
        config: Configuration dict

    Returns:
        Tokenized HuggingFace dataset
    """
    dataset = Dataset.from_list(formatted_data)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["data"]["max_length"],
            padding=False,  # Dynamic padding in DataCollator is more efficient
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    print("✓ Tokenization complete!")
    return tokenized_dataset


def main():
    """Main data preparation pipeline."""
    print("=" * 60)
    print("Rick Sanchez Fine-Tuning Data Preparation")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # 1. Load and process data from HuggingFace
    try:
        dialogues = load_and_process_data(max_retries=3)
    except Exception as e:
        print(f"\n❌ Critical Error: Failed to load data: {e}")
        print("Please check your internet connection and dataset availability.")
        return

    # 2. Load tokenizer
    print(f"\nLoading tokenizer: {config['model']['name']}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["name"], trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded!")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    # 3. Format data for training
    formatted_data = format_for_training(dialogues, tokenizer, config)

    if len(formatted_data) == 0:
        print("❌ No valid training examples after filtering. Aborting.")
        return

    print(f"\n📝 Sample training example:")
    print(f"   {formatted_data[0]['text'][:100]}...")

    # 4. Tokenize dataset
    tokenized_dataset = create_tokenized_dataset(formatted_data, tokenizer, config)

    # 5. Split into train/validation sets
    train_test_split_ratio = config["data"].get("train_split", 0.8)
    split_dataset = tokenized_dataset.train_test_split(
        test_size=1 - train_test_split_ratio, seed=42
    )

    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(split_dataset['train'])} examples")
    print(f"   Validation: {len(split_dataset['test'])} examples")

    # 6. Save to disk
    data_dir = Path(config["data"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving datasets to {data_dir}...")
    split_dataset["train"].save_to_disk(str(data_dir / "train"))
    split_dataset["test"].save_to_disk(str(data_dir / "val"))

    print("\n" + "=" * 60)
    print("✅ Data preparation complete! Ready for training.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the sample above to ensure quality")
    print("  2. Run: python finetune.py")
    print("  3. Or upload to Colab and run rick_training.ipynb")


if __name__ == "__main__":
    main()
