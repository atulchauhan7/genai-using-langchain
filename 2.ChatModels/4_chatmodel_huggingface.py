from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Who is lunatic coder? share it's youtube channel link.",
        }
    ],
)

print(completion.choices[0].message.content)