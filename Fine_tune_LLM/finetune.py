"""
Fine-tuning script for Rick Sanchez character model.
Uses LoRA + 4-bit quantization for efficient training on Google Colab.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import yaml
import os


class RickFineTuner:
    """Fine-tune a language model to speak like Rick Sanchez."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize fine-tuner with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with 4-bit quantization."""
        print(f"Loading model: {self.config['model']['name']}")
        
        # Configure 4-bit quantization
        if self.config['model'].get('use_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        print("Model and tokenizer loaded successfully!")
    
    def prepare_model_for_lora(self):
        """Prepare model for LoRA fine-tuning."""
        print("Preparing model for LoRA...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def load_datasets(self):
        """Load preprocessed training and validation datasets."""
        print("Loading datasets...")
        
        data_dir = self.config['data']['data_dir']
        
        self.train_dataset = load_from_disk(f"{data_dir}/train")
        self.val_dataset = load_from_disk(f"{data_dir}/val")
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments."""
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            eval_strategy="steps",
            save_total_limit=training_config['save_total_limit'],
            fp16=training_config['fp16'],
            optim=training_config['optim'],
            max_grad_norm=training_config['max_grad_norm'],
            load_best_model_at_end=True,
            report_to="none"  # Disable wandb/tensorboard for simplicity
        )
    
    def train(self):
        """Execute fine-tuning."""
        print("\n" + "="*50)
        print("Starting fine-tuning...")
        print("="*50 + "\n")
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        # Train
        trainer.train()
        
        print("\n" + "="*50)
        print("Training complete!")
        print("="*50 + "\n")
        
        return trainer
    
    def save_model(self, output_dir: str = None):
        """Save the fine-tuned model."""
        if output_dir is None:
            output_dir = self.config['training']['output_dir']
        
        print(f"Saving model to {output_dir}...")
        
        # Save LoRA adapter
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("Model saved successfully!")
    
    def run_full_pipeline(self):
        """Run complete fine-tuning pipeline."""
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare for LoRA
        self.prepare_model_for_lora()
        
        # Load datasets
        self.load_datasets()
        
        # Train
        trainer = self.train()
        
        # Save model
        self.save_model()
        
        return trainer


def main():
    """Main function to run fine-tuning."""
    
    print("Rick Sanchez Fine-Tuning")
    print("="*50)
    
    # Initialize fine-tuner
    finetuner = RickFineTuner(config_path="config.yaml")
    
    # Run full pipeline
    trainer = finetuner.run_full_pipeline()
    
    print("\nFine-tuning complete!")
    print("Model saved and ready for inference.")


if __name__ == "__main__":
    main()
