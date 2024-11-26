# Triton Inference Server

This repository contains all the model_repository and their config.pbtxt and model files to deploy the models using Triton Inference Server.

Launching and maintaining the Triton Inference Server revolves around the use of building model repositories.
. Creating a Model Repository
. Launching Triton
. Send an Inference Request
. Triton Inference Server has support for TensorFlow, PyTorch, TensorRT, ONNX and OpenVINO models.


# Get the docker container of triton inference server
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3 - choose the latest version. 

It mainly uses GPU and Nvidia Cuda for its working but if you don't have GPU then you can also work on it using CPU just choose the container compatible with CPU.You can choose the suitable container from the link below.
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags

# The structure of the folders of the model repository
```
model_repository/ 
└── <model-name>/ 
           ── config.pbtxt
          ── 1/ 
         	── model.onnx # or the appropriate model file (model.pt, model.pb etc)
                └── (other files if required) 
```
                   
# Example of config.pbtxt file

```
name: "<model-name>" ex - NuExtract
platform: "<runtime-name>" ex - onnxruntime_onnx
max_batch_size: 8
input [
  {
	name: "input_ids"
	data_type: TYPE_INT32
	dims: [ -1, -1 ]  # Dynamic batch size and sequence length
  }
]
output [
  {
	name: "logits"
	data_type: TYPE_FP32
	dims: [ -1, -1, -1 ]
  }
]
```
# Steps to Create config.pbtxt

**A. Understand Model Inputs and Outputs**

Obtain model input/output details:
For a Hugging Face/Transformers model, inputs typically include:
input_ids (token IDs)
attention_mask (binary mask)
Optional: past_key_values (cache)

Outputs might include:

output_ids (generated tokens)
logits (probabilities of the next token).

Example for your llama7b model:

Inputs:
input_ids: Token IDs (integer sequence).
attention_mask: Binary mask (same length as input_ids).

Outputs:
output_ids: Sequence of generated tokens.

**B. Model-Specific Configuration**

Data Types: Match your input/output types:
Token IDs (input_ids) → TYPE_INT32.
Masks (attention_mask) → TYPE_INT32.
Outputs (output_ids) → TYPE_INT32.

Shapes:
Use [-1] for dynamic lengths.
For fixed dimensions, specify exact sizes.

**C. Choose the Correct Platform**

Use "python" if you are running custom Python code for inference (e.g., with Transformers pipeline or PyTorch).
If using a pre-converted ONNX model, use "onnxruntime_onnx".

**D. Specify Hardware Resources**

For NVIDIA GPU-enabled devices, set:
instance_group [
  {
    kind: KIND_GPU
  }]
Otherwise, use:

instance_group [
  {
    kind: KIND_CPU
  }
]

# Script to create a model.onnx file
```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
model_name = "<model-name >" example - numind/NuExtract-1.5-smol
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
 
**Dummy input for export**
dummy_input = tokenizer("Example input text", return_tensors="pt").input_ids
 
**Export model to ONNX**
torch.onnx.export(
	model,
	(dummy_input,),
    "model.onnx",
    opset_version=13,  # Adjust opset version if required
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "logits": {0: "batch_size", 1: "sequence_length"}},
)
print("Model successfully exported to ONNX!")
 ```

# Script for detecting the model architecture
You need to know the model architecture in order to create a config.pbtxt file. This file will give ou the architecture of the document from which you can extract the input and outputs name and sizes and much more.
```
import argparse
import os
import onnx
import torch
import tensorflow as tf
import numpy as np
try:
    from openvino.runtime import Core
    from openvino.runtime.passes import Manager
except ImportError:
    Core = None
    Manager = None
try:
    import tensorrt as trt
except ImportError:
    trt = None
def inspect_onnx_model(model_path):
    model = onnx.load(model_path)
    print("\nONNX Model Inspection:")
    print("Inputs:")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  - Name: {input_tensor.name}, Shape: {shape}, Type: {input_tensor.type.tensor_type.elem_type}")
    print("Outputs:")
    for output_tensor in model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  - Name: {output_tensor.name}, Shape: {shape}, Type: {output_tensor.type.tensor_type.elem_type}")

def inspect_pytorch_model(model_path):
    model = torch.load(model_path, map_location=torch.device("cpu"))
    print("\nPyTorch Model Inspection:")
    print(model)

def inspect_tensorflow_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print("\nTensorFlow Model Inspection:")
    print(model.summary())

def inspect_tensorrt_model(model_path):
    if trt is None:
        print("TensorRT support is not available.")
        return
    print("\nTensorRT Model Inspection:")
    logger = trt.Logger(trt.Logger.WARNING)
    with open(model_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = engine.get_binding_dtype(i)
        io_type = "Input" if engine.binding_is_input(i) else "Output"
        print(f"  - {io_type} Name: {name}, Shape: {shape}, Type: {dtype}")

def inspect_openvino_model(model_path):
    if Core is None:
        print("OpenVINO support is not available.")
        return
    core = Core()
    model = core.read_model(model_path)
    print("\nOpenVINO Model Inspection:")
    print("Inputs:")
    for input_tensor in model.inputs:
        print(f"  - Name: {input_tensor.any_name}, Shape: {input_tensor.shape}, Type: {input_tensor.element_type}")
    print("Outputs:")
    for output_tensor in model.outputs:
        print(f"  - Name: {output_tensor.any_name}, Shape: {output_tensor.shape}, Type: {output_tensor.element_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect various model formats.")
    parser.add_argument("model_path", help="Path to the model file")
    parser.add_argument("--framework", choices=["onnx", "pytorch", "tensorflow", "tensorrt", "openvino"],
                        required=True, help="Framework of the model")
    args = parser.parse_args()

    if args.framework == "onnx":
        inspect_onnx_model(args.model_path)
    elif args.framework == "pytorch":
        inspect_pytorch_model(args.model_path)
    elif args.framework == "tensorflow":
        inspect_tensorflow_model(args.model_path)
    elif args.framework == "tensorrt":
        inspect_tensorrt_model(args.model_path)
    elif args.framework == "openvino":
        inspect_openvino_model(args.model_path)
    else:
        print("Unsupported framework!")
```
python3 inspect_model.py path/to/model.onnx --framework <framework-name>

# Example of a model.pt file
```
import os
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton_python_backend_utils as pb_utils
 
class TritonPythonModel:
	def initialize(self, args):
    	**Set up the cache directory for Hugging Face**
        os.environ["TRANSFORMERS_CACHE"] = "/opt/tritonserver/model_repository/smolLM2/hf-cache"
    	**Load model configuration from Triton server**
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
 
    	**Define model path**
    	default_model = "HuggingFaceTB/SmolLM2-1.7B"
    	model_name = self.model_params.get("huggingface_model", {}).get("string_value", default_model)
 
    	**Define max output length with a default value**
        default_max_length = "20"
        self.max_output_length = int(self.model_params.get("max_output_length", {}).get("string_value", default_max_length))
 
    	# Initialize tokenizer and model
    	self.logger = pb_utils.Logger
        self.logger.log_info(f"Loading HuggingFace model: {model_name} with max output length: {self.max_output_length}")
 
    	# Load tokenizer and model
    	self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    	self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode
 
	def execute(self, requests):
    	responses = []
    	for request in requests:
        	# Parse inputs
        	input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()
        	# Convert inputs to PyTorch tensors
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
        	# Generate outputs
        	with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                    max_length=self.max_output_length
            	)
        	# Convert outputs to numpy array for Triton
        	output_ids = outputs.cpu().numpy()
 
        	# Create response output tensor
            output_tensor = pb_utils.Tensor("output_ids", output_ids)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
    	return responses
	def finalize(self):
     self.logger.log_info("Finalizing the SmolLM2 model.")
 ```

# Run the container of the server with CPU and allocate the ports
```
 docker run -it --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
-v ${PWD}/model_repository:/opt/tritonserver/model_repository \
triton-transformer-server tritonserver --model-repository=/opt/tritonserver/model_repository
```
# Client-Side Inference
You can use the Triton Client SDK to send inference requests to the server.
Install Triton Client SDK:

pip install tritonclient[all]

**Example Python Script for Inference:**
```
import numpy as np
import tritonclient.http as httpclient  # Use grpcclient for gRPC

** Define server URL**
TRITON_SERVER_URL = "localhost:8000"

**Initialize client**
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

** Model name**
model_name = "mymodel"

** Check server health**
if not client.is_server_live():
    raise RuntimeError("Triton server is not live!")

** Generate a dummy input for testing**
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

**Create inference input**
input_tensor = httpclient.InferInput("input_0", input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

**Prepare inference output**
output_tensor = httpclient.InferRequestedOutput("output_0")

**Perform inference**
response = client.infer(model_name, inputs=[input_tensor], outputs=[output_tensor])

**Extract output**
output_data = response.as_numpy("output_0")
print("Model output:", output_data)
```
# Link of Triton Inference Server Documentation
 
https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html


