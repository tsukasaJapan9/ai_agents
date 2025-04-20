import json

import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict

TEMPERATUE = 0.5

API_URL = "http://localhost:8000/infer"

def get_system_messages() -> SystemMessage:
  return SystemMessage(content="あなたは優秀なAIエージェントです。気さくで楽しく明るい性格で、ユーザの入力に対してユーモラスに返答します。")

def main() -> None:
  # UIの初期化
  st.set_page_config(
    page_title="Chat App",
  )
  st.header("Chat app using MCP")

  # ここはあとから修正
  # 性格をPOSTして切り替えられると面白そう
  if "messages" not in st.session_state:
    st.session_state.messages = [get_system_messages()]

  if user_input := st.chat_input("何でも入力してね！"):
    # 会話履歴はlangchainのオブジェクトで管理する
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.spinner("AI agent is typing..."):
      # 会話履歴 + ユーザの入力をjsonにしてAPIで推論サーバにpost
      print("=================================")
      print(f"---------- [UI]: send data to infer server ----------")
      for message in st.session_state.messages:
        print(f"{message.__class__.__name__}: {message.content}")

      json_data = messages_to_dict(st.session_state.messages)
      json_str = json.dumps(json_data, indent=2)
      response = requests.post(API_URL, json={"message": json_str})
      json_data = response.json()
      json_data = json.loads(json_data["response"])
      
    # 推論サーバからデータがjsonで来るのでlangchainのオブジェクトに変換
    recevied_msgs = messages_from_dict(json_data)
    print(f"---------- [UI]: receved data from infer server ----------")
    
    # 推論サーバからは履歴も含めて送られてくるので履歴を初期化する
    st.session_state.messages = []
    for message in recevied_msgs:
      st.session_state.messages.append(message)
      print(f"{message.__class__.__name__}: {message.content}")

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
