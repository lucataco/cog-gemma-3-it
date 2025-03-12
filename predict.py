# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_CACHE = "checkpoints"
# Choose a model size
MODEL_URL = "https://weights.replicate.delivery/default/google/gemma-3-4b-it/model.tar"
# MODEL_URL = "https://weights.replicate.delivery/default/google/gemma-3-12b-it/model.tar"
# MODEL_URL = "https://weights.replicate.delivery/default/google/gemma-3-27b-it/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        # Load model with bfloat16 precision on GPU
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_CACHE, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        
        # Load processor for tokenization and image processing
        # Explicitly use the slow processor since the fast one has a bug?
        self.processor = AutoProcessor.from_pretrained(
            MODEL_CACHE,
            use_fast=False
        )

    def predict(
        self,
        prompt: str = Input(description="Text prompt for the model"),
        image: Path = Input(description="Optional image input for multimodal tasks", default=None),
        system_prompt: str = Input(description="System prompt to guide the model's behavior", default="You are a helpful assistant."),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=512, ge=1, le=8192),
        temperature: float = Input(description="Sampling temperature", default=0.7, ge=0.0, le=2.0),
        top_p: float = Input(description="Top-p sampling", default=0.9, ge=0.0, le=1.0),
        top_k: int = Input(description="Top-k sampling", default=50, ge=0, le=100),
    ) -> str:
        """Run a single prediction on the model"""
        
        # Prepare messages for the chat template
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": []
            }
        ]
        
        # Add image to user message if provided
        if image is not None:
            # Open the image and convert to RGB
            img = Image.open(image).convert("RGB")
            # Use the image directly in the message
            messages[1]["content"].append({"type": "image", "image": img})
        
        # Add text prompt to user message
        messages[1]["content"].append({"type": "text", "text": prompt})
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        # Get the length of the input to extract only the generated tokens later
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate text
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None
            )
            # Extract only the generated tokens
            generation = generation[0][input_len:]
        
        # Decode the generated tokens
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return decoded
