"""
Inference script for Rick Sanchez character model.
Load the fine-tuned model and generate responses in Rick's voice.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import sys
import yaml
import re


class RickInference:
    """Generate text in Rick Sanchez's voice."""

    def __init__(self, base_model_name: str, adapter_path: str):
        """
        Initialize inference engine.

        Args:
            base_model_name: Name of base model (e.g., "gpt2-medium")
            adapter_path: Path to fine-tuned LoRA adapter
        """
        print(f"Loading base model: {base_model_name}...", flush=True)

        # Verify adapter path exists
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter path not found: {adapter_path}\n"
                f"Make sure you've trained the model first using finetune.py"
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        print(f"Loading adapter from: {adapter_path}...", flush=True)

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval()

        # Optimize for inference if possible
        if hasattr(torch, "compile") and device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✓ Model compiled for faster inference")
            except Exception:
                pass  # Compilation not critical

        print("🧪 Wubba Lubba Dub Dub! Model loaded successfully!", flush=True)

    def generate_response(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.92,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
    ) -> str:
        """
        Generate a response as Rick Sanchez.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.7 = balanced)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetitive text

        Returns:
            Generated response
        """
        # Format prompt as a dialogue script to match training data
        # Training format was: "Rick Sanchez: {text}"
        # We simulate a conversation where Morty asks and Rick responds
        full_prompt = f"Morty: {prompt}\n" f"Rick Sanchez:"

        # Tokenize
        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.base_model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length + len(inputs["input_ids"][0]),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,  # Avoid exact phrase repetition
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response - everything after "Rick Sanchez:"
        if "Rick Sanchez:" in generated_text:
            parts = generated_text.split("Rick Sanchez:")
            response = parts[-1].strip()
        else:
            response = generated_text

        # Clean up the response
        response = self._clean_response(response)

        return response

    def _clean_response(self, response: str) -> str:
        """
        Clean up generated response by removing artifacts.

        Args:
            response: Raw generated text

        Returns:
            Cleaned response
        """
        # Remove text in brackets [], parentheses (), braces {}, angle brackets <>
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)
        response = re.sub(r"\{.*?\}", "", response)
        response = re.sub(r"\<.*?\>", "", response)

        # Remove speaker labels if they appear
        response = re.sub(r"Rick Sanchez:|Rick:|Morty:", "", response)

        # Collapse multiple spaces and newlines
        response = re.sub(r"\s+", " ", response).strip()

        # Stop at natural ending if response is too long
        # Look for sentence-ending punctuation
        sentences = re.split(r"([.!?])\s+", response)
        if len(sentences) > 1:
            # Keep reasonable number of sentences (3-5)
            max_sentences = 5
            cleaned_sentences = []
            for i in range(0, min(len(sentences), max_sentences * 2), 2):
                if i + 1 < len(sentences):
                    cleaned_sentences.append(sentences[i] + sentences[i + 1])
                else:
                    cleaned_sentences.append(sentences[i])
            response = " ".join(cleaned_sentences).strip()

        return response

    def interactive_chat(self):
        """Interactive chat interface."""
        print("\n" + "=" * 60)
        print("Rick Sanchez Interactive Chat - *Burp* 🧪")
        print("=" * 60)
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'help' for generation tips")
        print("=" * 60 + "\n")

        conversation_history = []

        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nFine, leave! I have better things to do anyway.")
                break

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nFine, leave! I have better things to do anyway.")
                break

            if user_input.lower() == "help":
                print("\n💡 Tips for better responses:")
                print("  - Ask Rick about science, inventions, or adventures")
                print("  - Reference Morty, the family, or interdimensional travel")
                print("  - Keep questions conversational")
                print("  - The model works best with direct questions\n")
                continue

            if not user_input:
                continue

            # Generate response
            print("\nRick: ", end="", flush=True)
            try:
                response = self.generate_response(user_input)
                print(response)
                conversation_history.append((user_input, response))
            except Exception as e:
                print(f"Error generating response: {e}")
                print("Try a different prompt or restart the chat.")
            print()


def load_config():
    """Load configuration from `config.yaml` or `config_fixed.yaml` if present.

    Returns:
        dict or None: Parsed YAML configuration, or `None` if not found.
    """
    # Check cwd first, then script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for cfg in ("config.yaml", "config_fixed.yaml"):
        if os.path.exists(cfg):
            with open(cfg, "r") as f:
                return yaml.safe_load(f)
        alt = os.path.join(script_dir, cfg)
        if os.path.exists(alt):
            with open(alt, "r") as f:
                return yaml.safe_load(f)
    return None


def get_adapter_path():
    """Get adapter path from various sources."""
    # Priority 1: Command-line argument
    if len(sys.argv) > 1:
        return sys.argv[1]

    # Priority 2: config.yaml
    config = load_config()
    if config:
        return config["training"]["output_dir"]

    # Priority 3: Default path
    return "./rick-sanchez-model"


def main():
    """Main function for inference."""

    print("=" * 60)
    print("Rick Sanchez Model - Inference")
    print("=" * 60 + "\n")

    # Get model paths
    config = load_config()
    base_model = config["model"]["name"] if config else "gpt2-medium"
    adapter_path = get_adapter_path()

    print(f"Base model: {base_model}")
    print(f"Adapter path: {adapter_path}\n")

    # Initialize inference
    try:
        rick = RickInference(base_model_name=base_model, adapter_path=adapter_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo train the model first, run:")
        print("  python data_preparation.py")
        print("  python finetune.py")
        return
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test generation
    print("\n" + "=" * 60)
    print("Testing Rick Sanchez Model")
    print("=" * 60 + "\n")

    test_prompts = [
        "Hi Rick, what are you working on?",
        "Can you explain quantum mechanics?",
        "Wubba Lubba Dub Dub!",
    ]

    for prompt in test_prompts:
        print(f"You: {prompt}")
        try:
            response = rick.generate_response(prompt)
            print(f"Rick: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

    print("=" * 60 + "\n")

    # Start interactive chat
    rick.interactive_chat()


if __name__ == "__main__":
    main()
