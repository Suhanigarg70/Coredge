import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the model and tokenizer
model_name = "microsoft/Phi-3.5-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dummy input to trace the model
dummy_input = tokenizer("This is a test sentence.", return_tensors="pt").input_ids

# Export the model to ONNX
onnx_output_path = "model.onnx"
torch.onnx.export(
    model,
    (dummy_input,),  # Model inputs
    onnx_output_path,  # Output ONNX file
    opset_version=14,  # Specify the ONNX opset version
    input_names=["input_ids"],  # Input tensor name
    output_names=["logits"],  # Output tensor name
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},  # Allow dynamic batch size and sequence length
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    do_constant_folding=True,  # Optimize for constant folding
    use_external_data_format=True  # Save large tensors as external data
)

print("Model has been successfully exported to ONNX format!")

