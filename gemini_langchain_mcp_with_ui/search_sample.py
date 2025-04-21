import os

import requests

GOOGLE_CUSTOM_SEARCH_API_KEY = os.environ["GOOGLE_CUSTOM_SEARCH_API_KEY"]
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.environ["GOOGLE_CUSTOM_SEARCH_ENGINE_ID"]


def google_custom_search(query, api_key, search_engine_id, num_results=3):
  url = "https://www.googleapis.com/customsearch/v1"
  params = {"key": api_key, "cx": search_engine_id, "q": query, "num": num_results}

  response = requests.get(url, params=params)
  if response.status_code != 200:
    raise Exception(f"Error: {response.status_code} - {response.text}")

  results = response.json().get("items", [])
  top_results = []
  for item in results:
    top_results.append({"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")})

  return top_results


query = "ChatGPT 使い方"

results = google_custom_search(query, GOOGLE_CUSTOM_SEARCH_API_KEY, GOOGLE_CUSTOM_SEARCH_ENGINE_ID)
for i, result in enumerate(results, 1):
  print(f"{i}. {result['title']}")
  print(f"URL: {result['link']}")
  print(f"概要: {result['snippet']}\n")
  print(f"概要: {result['snippet']}\n")
