import json

import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict

# TODO: ruffが効かない

TEMPERATUE = 0.5
SEND_MSG_SIZE = 10

INF_SERVER_URL = "http://localhost:8000"

personality_options: dict[str, str] = {
  "funny": "あなたは優秀なAIエージェントです。気さくで楽しく明るい性格で、ユーザの入力に対してユーモラスに返答してください。",
  "cool": "あなたは超天才AIエージェントです。ユーザの質問に的確に性格に分かりやすくクールに返答してください。",
  "crazy": "あなたはポンコツダメダメAIエージェントです。ユーザの質問に対して面白おかしくクレイジーな方法で返答してください。",
}

def get_system_message(personality: str) -> SystemMessage:
  return SystemMessage(
    content=personality_options[personality],
    id=0,
  )

def main() -> None:
  # UIの初期化
  st.set_page_config(
    page_title="Chat App",
  )
  st.title("Chat app using MCP")

  # toolとresorceの取得
  response = requests.get(INF_SERVER_URL + "/tools")
  json_data = response.json()
  json_data = json.loads(json_data["tools"])

  st.subheader("Available tools")
  for name, desc in json_data.items():
    st.write("-------------------------")
    st.write(f"{name}: {desc}")
  st.write("-------------------------")

  st.subheader("Chat history")

  # サイドバー
  st.sidebar.title("オプション")
  selected_personality = st.sidebar.radio("性格を選んでね:", tuple(personality_options.keys()))
  if "personality" not in st.session_state or st.session_state.personality != selected_personality:
    st.session_state.personality = selected_personality
    st.session_state.messages = []

  # ここはあとから修正
  # 性格をPOSTして切り替えられると面白そう
  if "messages" not in st.session_state:
    st.session_state.messages = [get_system_message(selected_personality)]

  if user_input := st.chat_input("何でも入力してね！"):
    # 会話履歴はlangchainのオブジェクトで管理する
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("AI agent is typing..."):
      # 会話履歴 + ユーザの入力をjsonにしてAPIで推論サーバにpost
      print("=================================")

      # すべての履歴を送ると推論に時間がかかるので直近の数個を送る
      # また先頭にid=0のsystem messageが無い場合は付与する
      messages = st.session_state.messages[-SEND_MSG_SIZE:]
      if not isinstance(messages[0], SystemMessage) and messages[0].id != 0:
        messages.insert(0, get_system_message(selected_personality))

      print(f"---------- [UI]: send data to infer server ----------")
      for message in messages:
        print(f"{message.__class__.__name__}: {message.content}")

      json_data = messages_to_dict(messages)
      json_str = json.dumps(json_data, indent=2)
      response = requests.post(INF_SERVER_URL + "/infer", json={"message": json_str})
      json_data = response.json()
      json_data = json.loads(json_data["response"])

    # 推論サーバからデータがjsonで来るのでlangchainのオブジェクトに変換
    recevied_msgs = messages_from_dict(json_data)
    print(f"---------- [UI]: receved data from infer server ----------")

    # 推論サーバからは履歴も含めて送られてくるので履歴を初期化する
    st.session_state.messages = []
    for message in recevied_msgs:
      st.session_state.messages.append(message)
      print(f"{message.__class__.__name__}: {message.content}(id: {message.id})")

    # チャット履歴の描画
    messages = st.session_state.get("messages", [])
    for message in messages:
      content = message.content
      if isinstance(message, AIMessage):
        with st.chat_message("Assistant"):
          st.markdown(content)
      elif isinstance(message, HumanMessage):
        with st.chat_message("User"):
          st.markdown(content)
      else:
        with st.chat_message("system"):
          st.markdown(content)

if __name__ == "__main__":
  main()
