from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "HuggingFaceTB/SmolLM2-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Create dummy input for the model (adjust depending on your input format)
dummy_input = tokenizer("Hello, world!", return_tensors="pt")

# Export the model to ONNX
torch.onnx.export(model,
                  (dummy_input['input_ids'],),
                  "smolLM2.onnx",
                  input_names=["input_ids"],
                  output_names=["output"],
                  opset_version=12)

