# Training Guide: Rick Sanchez Model on Google Colab

This guide explains how to fine-tune the Rick Sanchez model using your GitHub repository and Google Colab (Free Tier).

## Prerequisites

1.  **Google Account**: To access Google Colab.
2.  **Hugging Face Account**: To access base models (like Llama-3 or GPT-2) and push your new model.
3.  **GitHub Account**: Where your code lives.
4.  **Hugging Face Token**: [Get one here](https://huggingface.co/settings/tokens) (Write permissions).

## Step 1: Push Your Code to GitHub

First, make sure your latest code (including `data_preparation.py` and `finetune.py`) is on GitHub.

```bash
git add .
git commit -m "Ready for training"
git push origin main
```

## Step 2: Open Google Colab

1.  Go to [colab.research.google.com](https://colab.research.google.com/).
2.  Click **New Notebook**.
3.  Go to **Runtime** > **Change runtime type**.
4.  Select **T4 GPU** (Hardware accelerator). This is free and sufficient for this task.

## Step 3: Setup Environment in Colab

Copy and paste the following code blocks into your Colab notebook cells.

### Cell 1: Clone Repository
Replace `YOUR_GITHUB_USERNAME/YOUR_REPO_NAME` with your actual repo details.

```python
!git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
!pip install -r requirements.txt
!pip install datasets peft bitsandbytes trl
```

### Cell 2: Login to Hugging Face
This is required to access models and push your results.

```python
from huggingface_hub import notebook_login
notebook_login()
```

## Step 4: Prepare Data

Run your data preparation script. This will download the dataset and format it for the model.

```python
!python data_preparation.py
```

*Note: If you encounter specific data errors, you might need to adjust `config.yaml` in the Colab file editor.*

## Step 5: Start Training

Run the fine-tuning script.

```python
!python finetune.py
```

This process will take:
- **~20-40 minutes** on a T4 GPU (depending on dataset size).
- Watch the loss curve—it should go down over time.

## Step 6: Save and Push Model

Once training is complete, your adapter files will be in the output directory (usually specified in `config.yaml`).

To upload your new model to Hugging Face:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load your new adapter
adapter_path = "./rick-sanchez-model" # Check your config.yaml for exact path
base_model_name = "gpt2-medium" # Or whatever base model you used

model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Push to Hub
my_repo = "YOUR_HF_USERNAME/rick-sanchez-v2"
model.push_to_hub(my_repo)
tokenizer.push_to_hub(my_repo)

print(f"Model pushed to {my_repo}")
```

## Step 7: Use Your Model

Now you can use this model in your local `inference_rick.py` script!

1.  Edit `inference_rick.py`.
2.  Change `adapter_path` to your Hugging Face repo ID (e.g., `ninadoinjashvili/rick-sanchez-v2`).
3.  Run it!
