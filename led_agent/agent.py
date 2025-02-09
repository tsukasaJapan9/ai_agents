import json
import os

import cv2
import numpy as np

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

MATRIX_WIDTH = 12
MATRIX_HEIGHT = 12
LED_SIZE = 50

PANEL_WIDTH = MATRIX_WIDTH * LED_SIZE
PANEL_HEIGHT = MATRIX_HEIGHT * LED_SIZE

SYSTEM_INSTRUCTION = \
f"""
あなたはLEDパネルを持っているロボットです。LEDパネルを使って感情を表現します。
LEDパネルは{MATRIX_HEIGHT}x{MATRIX_WIDTH}のマトリクスになっており、それぞれのLEDはBGRの3ch、それぞれ0 ~ 255の値を用いることができます。
こちらからのメッセージに対してLEDをどう光らせるかを決定してください。また点灯パターンは左右対象である必要があります。
なぜそのように考えたかの理由とLEDの点灯パターンをjsonで返してください。
"""
SYSTEM_INSTRUCTION += \
f"""
jsonの形式は以下の通りです。led_matrixは{MATRIX_HEIGHT}x{MATRIX_WIDTH}のマトリクスで、各要素は[R, G, B]の3要素のリストです。
led_matrixの縦と横の長さはそれぞれ{MATRIX_HEIGHT}と{MATRIX_WIDTH}です。
"""
SYSTEM_INSTRUCTION += \
"""
以下が、jsonのサンプルです。led_matrixの縦と横のサイズは必要に応じて変更してください。
responseはjsonのみで返してください。
{
  "reason": "悲しみ表現するために、青色を基調とし、眉を下げたような形にLEDを点灯させました。これにより、悲しんでいる印象を与えます。",
  "led_matrix": [
    [ [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 50], ... ],
    [ [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50], ... ],
    [ [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], ... ],
    [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ... ],
    [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ... ],
    [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ... ],
    [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ... ],
    [ [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], ... ],
    [ [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50], ... ],
    [ [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 50], ... ],
    ...
  ]
}
"""

# CONTENTS = "あなたは眼の前でとてもかわいい犬を見かけました。その時の感情を表現してください。"
CONTENTS = "あなたはテレビで戦争のニュースを見ました。その時の感情を表現してください。"
response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION),
    contents=CONTENTS,
)
response = response.text

# response = \
# """
# {
#   "reason": "悲しみを表現するために、青色を基調とし、眉を下げたような形にLEDを点灯させました。これにより、落ち込んだ印象を与えます。",
#   "led_matrix": [
#     [ [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 50] ],
#     [ [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50] ],
#     [ [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50] ],
#     [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
#     [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
#     [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
#     [ [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0] ],
#     [ [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50] ],
#     [ [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50] ],
#     [ [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 50], [0, 0, 50], [0, 0, 50], [0, 0, 50] ]
#   ]
# }
# """

if "```json" in response:
  response = response.replace("```json", "")
  response = response.replace("```", "")

print("-----------------")
print(CONTENTS)
print("-----------------")
print(response)
print("-----------------")

# responseをパースしてopencvで10x10の円を描画。色はjsonのled_matrixを参照
led_matrix = json.loads(response)['led_matrix']

# 空の画像を作成（黒で初期化）
panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

# # 画素の色を指定（例: 赤色）
# color = [0, 0, 255]  # BGR形式

# 各画素を円で描画
for y in range(MATRIX_HEIGHT):
  for x in range(MATRIX_WIDTH):
    # 何故かgemniniがRGBで返してくるので、BGRに変換
    led_matrix[y][x] = led_matrix[y][x][::-1]
    color = led_matrix[y][x]
    center = (x * LED_SIZE + LED_SIZE // 2, y * LED_SIZE + LED_SIZE // 2)
    cv2.circle(panel, center, LED_SIZE // 2, color, -1)

# 画像を表示
cv2.imshow('LED Panel', panel)
cv2.waitKey(0)
cv2.destroyAllWindows()
