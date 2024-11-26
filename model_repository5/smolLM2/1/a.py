import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Save the model in PyTorch format
model_path = "model.pt"
torch.save(model.state_dict(), model_path)

print(f"Model has been saved as {model_path}")

