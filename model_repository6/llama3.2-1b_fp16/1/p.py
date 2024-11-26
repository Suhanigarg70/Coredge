import onnx

def inspect_onnx_model(model_path):
    model = onnx.load(model_path)
    print("Model Input Info:")
    for input_tensor in model.graph.input:
        print(f"Name: {input_tensor.name}, Type: {input_tensor.type.tensor_type}, Shape: {input_tensor.type.tensor_type.shape.dim}")

    print("\nModel Output Info:")
    for output_tensor in model.graph.output:
        print(f"Name: {output_tensor.name}, Type: {output_tensor.type.tensor_type}, Shape: {output_tensor.type.tensor_type.shape.dim}")

model_path = "model.onnx"
inspect_onnx_model(model_path)

