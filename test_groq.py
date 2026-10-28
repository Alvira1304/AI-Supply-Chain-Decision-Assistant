
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Find the folder containing this Python file
project_folder = Path(__file__).resolve().parent
env_file = project_folder / ".env"

# Load environment variables from that file
load_dotenv(dotenv_path=env_file, override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        f"API key not found. Check that your .env file exists at: {env_file}"
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
   model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain low inventory in two simple sentences."
        }
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)
