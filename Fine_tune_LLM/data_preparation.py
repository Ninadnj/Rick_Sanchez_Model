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
STAGE_VERBS = (
    "stumbles",
    "rubs",
    "spills",
    "drags",
    "pulls",
    "pushes",
    "kicking",
    "jumps",
    "walks",
    "runs",
    "opens",
    "closes",
    "lands",
    "falls asleep",
    "snoring",
    "glaring",
    "tears up",
    "wipes",
    "stares",
    "throws",
    "presses",
    "turns on",
    "begins",
)
STAGE_VERB_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in STAGE_VERBS) + r")\b",
    flags=re.IGNORECASE,
)
SPEECH_HINT_RE = re.compile(
    r"\b(i|i'm|i've|i'll|you|you're|we|we're|what|why|how|can't|don't|won't|let's)\b",
    flags=re.IGNORECASE,
)
SPEAKER_NAME_RE = r"[A-Z][A-Za-z0-9'_-]{0,20}(?: [A-Za-z0-9'_-]{1,20}){0,5}"
SPEAKER_LABEL_RE = re.compile(SPEAKER_NAME_RE + r":{1,2}\s*")


def is_rick_speaker(name: str) -> bool:
    """Identify lines likely spoken by Rick (exclude Rick-owned devices/entities)."""
    if not isinstance(name, str):
        return False
    speaker = name.strip().lower()
    if not speaker:
        return False

    # Exclude things like "Rick's Car"
    if re.search(r"rick['’]s\b", speaker):
        return False

    # Common non-human/system entities that can include "Rick" in name
    if any(tag in speaker for tag in ("voice", "computer", "intercom", "robot", "ai")):
        return False

    return bool(re.search(r"\brick\b", speaker))


def _looks_like_stage_clause(sentence: str) -> bool:
    """
    Heuristic classifier for action/narration fragments in transcript text.
    """
    s = sentence.strip()
    if not s:
        return False

    # Example from dataset viewer: "stumbles in drunkenly, and turns on the lights."
    if re.match(r"^[a-z]", s) and STAGE_VERB_RE.search(s):
        return True

    # Action text often lacks first-person conversational tokens.
    if STAGE_VERB_RE.search(s) and not SPEECH_HINT_RE.search(s):
        return True

    # Multi-character scene notes
    if re.search(r"\b(at the same time|they begin to|starts [a-z]+ing)\b", s, flags=re.I):
        return True

    return False


def _drop_leading_stage_prefix(text: str) -> str:
    """
    Remove leading narration when a line starts with stage action before speech.
    """
    out = text.strip()
    for _ in range(3):
        if not out or not re.match(r"^[a-z]", out):
            break
        # Remove up to first sentence/scene delimiter then re-check.
        m = re.search(r"[\.\!\?\)]\s+", out)
        if not m:
            break
        prefix = out[: m.end()].strip()
        if _looks_like_stage_clause(prefix):
            out = out[m.end() :].strip()
            continue
        break
    return out


def _strip_speaker_label_noise(text: str) -> str:
    """
    Remove inline speaker labels and keep only Rick's first clean utterance.

    Examples:
      "Rick: Hi. Beth:: Hello." -> "Hi."
      "I'm Rick... Morty: huh?" -> "I'm Rick..."
    """
    out = text.strip()
    if not out:
        return ""

    # Remove a leading speaker tag if present.
    out = re.sub(r"^\s*" + SPEAKER_LABEL_RE.pattern, "", out).strip()
    if not out:
        return ""

    # Truncate at the next speaker label to avoid multi-speaker continuation.
    m = re.search(r"(\s+|\n)" + SPEAKER_LABEL_RE.pattern, out)
    if m:
        out = out[: m.start()].strip()
    return out


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
        
    # Remove bracketed stage directions [like this] or (like this)
    text = re.sub(r"[\[\(].*?[\]\)]", "", text)

    # Normalize whitespace early
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # Drop common leading narration before spoken content.
    text = _drop_leading_stage_prefix(text)

    # Filter per-sentence stage clauses while keeping speech.
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    kept = [s.strip() for s in sentences if s.strip() and not _looks_like_stage_clause(s)]
    text = " ".join(kept).strip()

    # Final whitespace normalization and punctuation cleanup
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;:.!?])", r"\1", text).strip()
    text = _strip_speaker_label_noise(text)
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

    # If cleaned text still looks like pure stage direction, reject
    if _looks_like_stage_clause(cleaned):
        return False
        
    # Check for remaining brackets that might have been missed or if line is just brackets
    if re.fullmatch(r"\s*[\[\(].*[\]\)]\s*", text):
        return False

    return True


def load_and_process_data(
    max_retries: int = 3,
    min_dialogue_length: int = 2,
    context_window_size: int = 3,
    use_single_turn_prompt: bool = True,
    require_question_prompt: bool = False,
):
    """
    Load Rick and Morty dataset and extract dialogue with context.

    Args:
        max_retries: Maximum number of retry attempts

    Returns:
        list[dict]: Records with "prompt" and "response"
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

            # Identify Rick's lines with extra protection against entities like "Rick's Car"
            df["is_rick"] = df["character"].apply(is_rick_speaker)

            # Fallback to broad match if strict heuristic finds nothing
            if df["is_rick"].sum() == 0:
                print("⚠️  Warning: No exact matches for 'Rick'. Trying broader matching...")
                df["is_rick"] = df["character"].str.contains(r"(?i)\brick\b", na=False)
                
            rick_count = df["is_rick"].sum()
            print(f"✓ Found {rick_count} lines potentialy spoken by Rick.")
            
            if rick_count < 50:
                raise ValueError(f"Insufficient data: only {rick_count} lines found.")

            # --- SUPERVISED PAIR GENERATION ---
            # Train on: Prompt ("User: ...\nRick Sanchez:") -> Response (Rick line)
            training_pairs = []

            episode_col = next(
                (c for c in df.columns if c.replace(" ", "").startswith("episode")),
                None,
            )
            
            def latest_non_rick_prompt(i: int, current_episode):
                """
                Return the nearest valid non-Rick utterance before index i.
                """
                for j in range(1, context_window_size + 1):
                    prev_idx = i - j
                    if prev_idx < 0:
                        break
                    if episode_col and df.iloc[prev_idx][episode_col] != current_episode:
                        break

                    prev_speaker = str(df.iloc[prev_idx]["character"]).strip()
                    if is_rick_speaker(prev_speaker):
                        continue

                    prev_text = str(df.iloc[prev_idx]["line"])
                    if not is_valid_dialogue(prev_text, min_length=min_dialogue_length):
                        continue

                    cleaned_prompt = clean_text(prev_text)
                    if cleaned_prompt:
                        return cleaned_prompt
                return ""

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
                if not cleaned_response:
                    continue
                
                current_episode = df.iloc[i][episode_col] if episode_col else None
                prompt_text = latest_non_rick_prompt(i, current_episode)
                if not prompt_text:
                    continue

                if require_question_prompt and "?" not in prompt_text:
                    continue

                # Keep prompt template consistent with inference.
                if use_single_turn_prompt:
                    prompt = f"User: {prompt_text}\nRick Sanchez:"
                else:
                    prompt = f"Morty: {prompt_text}\nRick Sanchez:"

                # Leading space keeps tokenization natural after colon
                response = f" {cleaned_response}"
                training_pairs.append({"prompt": prompt, "response": response})

            print(
                f"✓ Generated {len(training_pairs)} supervised examples with context."
            )
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


def create_tokenized_dataset(training_pairs, tokenizer, config):
    """
    Create and tokenize dataset.

    Args:
        training_pairs: List of {"prompt", "response"} records
        tokenizer: Tokenizer instance
        config: Configuration dict

    Returns:
        Tokenized HuggingFace dataset with masked labels
    """
    dataset = Dataset.from_list(training_pairs)
    max_length = int(config["data"].get("max_length", 512))

    def tokenize_function(examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for prompt, response in zip(examples["prompt"], examples["response"]):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(
                f"{response}{tokenizer.eos_token}", add_special_tokens=False
            )["input_ids"]

            input_ids = prompt_ids + response_ids
            labels = ([-100] * len(prompt_ids)) + response_ids

            # Keep the end of the sequence so response tokens are preserved.
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]

            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

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
            use_single_turn_prompt=config["data"].get("use_single_turn_prompt", True),
            require_question_prompt=config["data"].get("require_question_prompt", False),
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

    if len(training_pairs) == 0:
        print("❌ No valid training examples after filtering. Aborting.")
        return

    print("\n📝 Sample supervised training example:")
    print("-" * 40)
    print("Prompt:")
    print(training_pairs[0]["prompt"][:500])
    print("\nResponse:")
    print(training_pairs[0]["response"][:300])
    print("-" * 40)

    # 4. Tokenize dataset
    tokenized_dataset = create_tokenized_dataset(training_pairs, tokenizer, config)

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
