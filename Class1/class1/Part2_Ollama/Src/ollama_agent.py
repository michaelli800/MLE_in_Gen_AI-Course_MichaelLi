import ollama
import json

# Define available functions
def add_numbers(a, b):
    return a + b

def subtract_numbers(a, b):
    return a - b

def multiply_numbers(a, b):
    return a * b

# Helper: describe available functions for the model
function_descriptions = """
You can perform the following functions:

1. add_numbers(a, b): Adds two numbers.
2. subtract_numbers(a, b): Subtracts b from a.
3. multiply_numbers(a, b): a multiply b.

When you decide to call one, respond ONLY with a JSON object in this format:
{
  "function": "function_name",
  "arguments": { "a": number, "b": number }
}
"""

def get_agent_response(user_prompt, model="llama3.2"):
    messages = [
        {"role": "system", "content": "You are a reasoning assistant that can decide which function to call."},
        {"role": "assistant", "content": function_descriptions},
        {"role": "user", "content": user_prompt}
    ]

    # Query Llama through Ollama
    response = ollama.chat(model=model, messages=messages)
    content = response["message"]["content"].strip()
    print("Model response:", content)
    # Try to parse as JSON
    try:
        call = json.loads(content)
        func_name = call.get("function")
        args = call.get("arguments", {})

        if func_name == "add_numbers":
            result = add_numbers(**args)
        elif func_name == "subtract_numbers":
            result = subtract_numbers(**args)
        elif func_name == "multiply_numbers":
            result = multiply_numbers(**args)
        else:
            result = f"Unknown function: {func_name}"
        return result

    except json.JSONDecodeError:
        # Model didn’t return JSON; fallback
        return f"Model response: {content}"

# Example usage
if __name__ == "__main__":
    user_prompt = "What is 15 minus 7?"
    result = get_agent_response(user_prompt)
    print(result)
    user_prompt = "What is 15 + 7?"
    result = get_agent_response(user_prompt)
    print(result)
    user_prompt = "What is 15 * 7?"
    result = get_agent_response(user_prompt)
    print(result)
