
import numpy as np
import tritonclient.http as httpclient  # Use grpcclient for gRPC

# Define server URL
TRITON_SERVER_URL = "localhost:8000"

# Initialize client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Model name
model_name = "mymodel"

# Check server health
if not client.is_server_live():
    raise RuntimeError("Triton server is not live!")

# Generate a dummy input for testing
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Create inference input
input_tensor = httpclient.InferInput("input_0", input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

# Prepare inference output
output_tensor = httpclient.InferRequestedOutput("output_0")

# Perform inference
response = client.infer(model_name, inputs=[input_tensor], outputs=[output_tensor])

# Extract output
output_data = response.as_numpy("output_0")
print("Model output:", output_data)
