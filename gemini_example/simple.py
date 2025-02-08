import os
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def get_current_weather(location: str,) -> int:
  """Returns the current weather.

  Args:
    location: The city and state, e.g. San Francisco, CA
  """
  return 'sunny'

response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents="geminiを使ったAIガジェットを作りたい。案を10個。実装するためのデバイスやソフトウェアアーキテクチャも書いてください",
    config=types.GenerateContentConfig(tools=[get_current_weather],)
)

print(response.text)

# response = client.models.generate_content(model='gemini-2.0-flash-exp', contents='Can you speech Japanese?')
# response = client.models.generate_content(model='gemini-2.0-flash-exp', contents='日本語でお願いします。')
# print(response.text)


# import os
# import google.generativeai as genai

# API_KEY = os.environ.get("GENAI_API_KEY")

# genai.configure(api_key=API_KEY)
# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("Explain how AI works")
# print(response.text)