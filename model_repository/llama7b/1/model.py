import os
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        # Setting up the cache directory for Hugging Face
        os.environ["TRANSFORMERS_CACHE"] = "/opt/tritonserver/model_repository/llama3.2-1b/hf-cache"

        # Load model configuration from Triton server
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})

        # Define model path and token
        default_hf_model = "meta-llama/Llama-3.2-1B"
        hf_model = self.model_params.get("huggingface_model", {}).get("string_value", default_hf_model)
        private_repo_token = os.getenv("PRIVATE_REPO_TOKEN", "")

        # Define max output length with a default value
        default_max_gen_length = "20"
        self.max_output_length = int(self.model_params.get("max_output_length", {}).get("string_value", default_max_gen_length))

        # Initialize tokenizer and model
        self.logger = pb_utils.Logger
        self.logger.log_info(f"Loading HuggingFace model: {hf_model} with max output length: {self.max_output_length}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, use_auth_token=private_repo_token)
        self.model = AutoModelForCausalLM.from_pretrained(hf_model, use_auth_token=private_repo_token)
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

            # Run the model to get outputs
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
        self.logger.log_info("Finalizing the LLaMA 3.2 1B model.")

