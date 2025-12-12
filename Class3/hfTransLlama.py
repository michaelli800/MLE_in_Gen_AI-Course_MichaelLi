from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model name on Hugging Face
model_name = "meta-llama/Llama-3.2-1B-Instruct"  # or 3.2-3B, 3.2-8B, etc.

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,   # use float16 for efficiency
    device_map="auto"            # auto-loads on GPU if available
)

# Define prompt
prompt = "Explain quantum computing in simple terms."

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
