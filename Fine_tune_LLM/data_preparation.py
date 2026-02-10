#!/usr/bin/env python3
"""
Prepare Rick Sanchez (Rick and Morty) dialogue data for fine-tuning.
Downloads dataset from Hugging Face and formats for context-aware chat.
"""

import yaml
import time
import re
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Hugging Face Dataset ID
HF_DATASET_ID = "Prarabdha/Rick_and_Morty_Transcript"


def load_config():
    """Load configuration from `config.yaml` or fallback to `config_fixed.yaml`.

    Returns:
        tuple[dict, Path]: Parsed YAML configuration and resolved config file path.

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
                return yaml.safe_load(f), cfg.resolve()
        # 2) alongside script
        alt = script_dir / cfg
        if alt.exists():
            with open(alt, "r") as f:
                return yaml.safe_load(f), alt.resolve()
    raise FileNotFoundError(
        "No configuration file found. Create 'config.yaml' or 'config_fixed.yaml' in the project root or the script folder."
    )


def clean_text(text: str) -> str:
    """
    Clean text by removing stage directions and normalizing whitespace.
    
    Args:
        text: Raw dialogue text
        
    Returns:
        Cleaned text, or empty string if invalid
    """
    if not isinstance(text, str):
        return ""
        
    # Remove inline stage directions [like this] or (like this)
    text = re.sub(r"[\[\(].*?[\]\)]", "", text)
    
    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def is_valid_dialogue(text: str, min_length: int = 2) -> bool: # Reduced min_length to capture short replies like "No."
    """
    Filter out low-quality dialogue.

    Args:
        text: Dialogue text to validate
        min_length: Minimum character length

    Returns:
        True if dialogue is valid, False otherwise
    """
    cleaned = clean_text(text)

    # Too short
    if len(cleaned) < min_length:
        return False

    # Check if line is *entirely* stage directions (which clean_text might have emptied)
    if not cleaned:
        return False

    # Check for excessive asterisks (action descriptions)
    if cleaned.count("*") > 2:
        return False
        
    # Check for remaining brackets that might have been missed or if line is just brackets
    if re.fullmatch(r"\s*[\[\(].*[\]\)]\s*", text):
        return False

    return True


def load_and_process_data(
    max_retries: int = 3, min_dialogue_length: int = 2, context_window_size: int = 3
):
    """
    Load Rick and Morty dataset and extract dialogue with context.

    Args:
        max_retries: Maximum number of retry attempts

    Returns:
        list[str]: Context-aware training examples ending with a Rick response
    """
    for attempt in range(max_retries):
        try:
            print(f"Loading dataset {HF_DATASET_ID} from Hugging Face...")
            print(f"Attempt {attempt + 1} of {max_retries}")

            # Load dataset
            dataset = load_dataset(HF_DATASET_ID)

            # Convert to pandas for easier processing
            df = dataset["train"].to_pandas()

            print(f"✓ Loaded {len(df)} total lines.")
            print(f"Columns: {df.columns.tolist()}")

            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]

            # Map dataset columns to standard names
            if "speaker" in df.columns:
                df = df.rename(columns={"speaker": "character"})
            
            # Handle typo in dataset ('dialouge')
            if "dialogue" in df.columns:
                df = df.rename(columns={"dialogue": "line"})
            elif "dialouge" in df.columns:
                df = df.rename(columns={"dialouge": "line"})

            # Validate required columns exist
            if "character" not in df.columns or "line" not in df.columns:
                raise ValueError(
                    f"Expected 'character' and 'line' columns. "
                    f"Found: {df.columns.tolist()}"
                )

            # Clean character names
            df["character"] = df["character"].astype(str).str.strip()
            
            # Identify Rick's lines using regex
            # Matches "Rick", "Rick Sanchez", "Rick (C-137)", etc.
            df["is_rick"] = df["character"].str.fullmatch(r"(?i)rick(\s+sanchez)?.*", na=False)
            
            # Fallback if strict regex fails (though broad regex above should catch most)
            if df["is_rick"].sum() == 0:
                print("⚠️  Warning: No exact matches for 'Rick'. Trying broader matching...")
                df["is_rick"] = df["character"].str.contains(r"(?i)\brick\b", na=False)
                
            rick_count = df["is_rick"].sum()
            print(f"✓ Found {rick_count} lines potentialy spoken by Rick.")
            
            if rick_count < 50:
                raise ValueError(f"Insufficient data: only {rick_count} lines found.")

            # --- CONTEXT GENERATION ---
            # We want to train on: Context (Previous speakers) -> Response (Rick)
            training_pairs = []
            # Number of previous turns to include
            
            # Helper to format line
            def format_line(idx):
                speaker = df.iloc[idx]["character"]
                text = clean_text(str(df.iloc[idx]["line"]))
                return f"{speaker}: {text}"

            for i in range(len(df)):
                row = df.iloc[i]
                
                # We only want to train on RICK'S responses
                if not row["is_rick"]:
                    continue
                    
                # Get current line
                current_line_text = str(row["line"])
                
                # Check validity
                if not is_valid_dialogue(
                    current_line_text, min_length=min_dialogue_length
                ):
                    continue
                    
                cleaned_response = clean_text(current_line_text)
                target_response = f"Rick Sanchez: {cleaned_response}"
                
                # Build context from previous lines
                context_lines = []
                # Look back up to 'context_window_size' lines
                for j in range(1, context_window_size + 1):
                    prev_idx = i - j
                    if prev_idx < 0:
                        break
                    
                    # Check if previous line is valid
                    prev_text = str(df.iloc[prev_idx]["line"])
                    if not is_valid_dialogue(
                        prev_text, min_length=min_dialogue_length
                    ):
                        continue
                        
                    context_lines.insert(0, format_line(prev_idx))
                
                # Construct full prompt only if we have context
                # (Training on context-less lines is okay for style, but context is better)
                if context_lines:
                    full_prompt = "\n".join(context_lines) + "\n" + target_response
                else:
                    # Fallback for start of conversations
                    full_prompt = target_response
                    
                training_pairs.append(full_prompt)

            print(f"✓ Generated {len(training_pairs)} training examples with context.")
            return training_pairs

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("❌ All retry attempts exhausted.")
                raise


def format_for_training(training_pairs, tokenizer, config):
    """
    Format pairs for training with EOS token.

    Args:
        training_pairs: List of prompt strings
        tokenizer: Tokenizer instance
        config: Configuration dict

    Returns:
        List of formatted training examples
    """
    formatted_data = []
    
    for text in training_pairs:
        # Add EOS token to the end
        formatted_text = f"{text}{tokenizer.eos_token}"
        formatted_data.append({"text": formatted_text})

    print(f"✓ Final count: {len(formatted_data)} ready for tokenization.")
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
            max_length=config["data"].get("max_length", 512),
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
    config, config_file = load_config()
    project_dir = config_file.parent

    # 1. Load and process data from HuggingFace
    try:
        min_dialogue_length = config["data"].get("min_dialogue_length", 2)
        context_window_size = config["data"].get("context_window_size", 3)
        print(
            f"Using min_dialogue_length={min_dialogue_length}, "
            f"context_window_size={context_window_size}"
        )

        training_pairs = load_and_process_data(
            max_retries=3,
            min_dialogue_length=min_dialogue_length,
            context_window_size=context_window_size,
        )
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
        
        # Verify EOS token
        print(f"EOS Token: {tokenizer.eos_token}")
        
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    # 3. Format data for training
    formatted_data = format_for_training(training_pairs, tokenizer, config)

    if len(formatted_data) == 0:
        print("❌ No valid training examples after filtering. Aborting.")
        return

    print(f"\n📝 Sample training example (Context -> Response):")
    print("-" * 40)
    print(formatted_data[0]['text'][:500]) # Print first 500 chars
    print("-" * 40)

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
    if not data_dir.is_absolute():
        data_dir = (project_dir / data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving datasets to {data_dir}...")
    split_dataset["train"].save_to_disk(str(data_dir / "train"))
    split_dataset["test"].save_to_disk(str(data_dir / "val"))

    print("\n" + "=" * 60)
    print("✅ Data preparation complete! Ready for training.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review context pairs in the sample above")
    print("  2. Run: python finetune.py")
    print("  3. Or upload to Colab and run rick_training.ipynb")


if __name__ == "__main__":
    main()
