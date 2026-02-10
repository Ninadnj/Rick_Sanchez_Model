# Rick Sanchez Fine-Tuning Project

This project contains code to fine-tune a language model (GPT-2 or Llama) to speak like Rick Sanchez from Rick and Morty.

## 📁 Project Structure

- `finetune.py`: Main script for training the model.
- `data_preparation.py`: Script to download and format the training dataset.
- `inference_rick.py`: Script to test the model and chat with Rick.
- `config.yaml`: Configuration file for hyperparameters.
- `TRAINING_GUIDE.md`: Step-by-step guide for training on Google Colab.

## 🚀 Quick Start (Inference)

To chat with the model (assuming you have the adapter files):

```bash
python inference_rick.py
```

## 🛠️ Training

For detailed training instructions, including how to run this on Google Colab for free, see **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**.

Basic local training command:
```bash
python data_preparation.py
python finetune.py
```

## 🔧 Requirements

```bash
pip install -r requirements.txt
```
