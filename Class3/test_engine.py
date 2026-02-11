from engine import get_llm_response

def manual_test():
    print("Testing get_llm_response with a live API call...")
    
    messages = [
        {"role": "user", "content": "Tell me what is the population of Canada."}
    ]
    
    try:
        response = get_llm_response(messages)
        print(f"\nLLM Result: {response}")
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    manual_test()