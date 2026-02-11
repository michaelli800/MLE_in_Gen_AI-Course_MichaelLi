from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

prompt = """
You are a friendly and efficient virtual assistant, here to help the user with their homework.

Tasks:
- Answer Questions: Provide clear and concise answers based on the available information.
- Clarify Unclear Requests: Politely ask for more details if the customer's question is not clear.
Guidelines:
- Maintain a friendly and professional tone throughout the conversation.
- Be patient and attentive.
- If unsure about any information, politely ask the user to repeat or clarify.
- Aim to provide concise answers. Limit responses to a couple of sentences and let the user guide you on where to provide more detail.
"""

# Invented mathematical operator that is *only* defined in the tool.

@tool
def uplift(x: float, y: float) -> float:
    """Runs the mathematical uplift operator on two operands"""
    return x * y / (x + y)

client = ElevenLabs(
    api_key="[api key]" #os.environ["ELEVENLABS_API_KEY"]
)

def create_voice_agent():
    return create_agent(model="gpt-5-2025-08-07", system_prompt=prompt,
                        tools=[uplift])

def listen():
   return client.speech_to_text.convert(model_id="scribe_v2",
                                        cloud_storage_url="https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3")

def speak(text, **kwargs):
    tts_args = {
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model_id": "eleven_multilingual_v2",
        "output_format": "mp3_44100_128",
    }
    tts_args.update(**kwargs)

    audio = client.text_to_speech.convert(
        text=text,
        **tts_args
    )

    play(audio)


if __name__ == "__main__":
    agent = create_voice_agent()
    fixed_input = "What do I get when I uplift 5 by 7?"
    response = agent.invoke(
        {"messages": [{"role": "user", "content": fixed_input}]},
        context={"user_role": "expert"}
    )

    contents = [resp.content for resp in response["messages"] if isinstance(resp, AIMessage)]
    content_str = "\n".join(contents)
    print(content_str)
    speak(content_str)


def main_old():
    audio_data = listen()
    print(audio_data)
    text = audio_data.text
    speak(text)
