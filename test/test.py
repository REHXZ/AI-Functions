from google import genai
from google.genai import types
import os

def initialize_model(key="AIzaSyBYFZveb5j2EY6tbmSE6Bkyugi6pA8GPOk", model="gemini-2.5-flash"):
    client = genai.Client(
        api_key=key,
    )

    model = model
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="What is the capital of France?"),
            ],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch(
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        tools=tools,
    )

    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    return response_text
x = initialize_model()
print(x)