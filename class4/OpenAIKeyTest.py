import os
from openai import OpenAI
from dotenv import load_dotenv

# It's best practice to load the API key from an environment variable
# client = OpenAI() 


load_dotenv()  # This will load variables from the .env file into the environment
# Now you can access the API key
api_key1 = os.getenv("OPENAI_API_KEY")

# Or, you can pass it directly for a quick test:
client = OpenAI(api_key=api_key1)
try:
    # Attempt a simple API call, like listing models
    models = client.models.list()
    print("API key is valid. Models available:")
    for model in models.data:
        print(f"- {model.id}")
except openai.AuthenticationError:
    print("Error: Invalid OpenAI API key.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")