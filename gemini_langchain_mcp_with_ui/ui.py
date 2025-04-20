import json

import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict

TEMPERATUE = 0.5

API_URL = "http://localhost:8000/infer"

def main() -> None:
  # UIの初期化
  st.set_page_config(
    page_title="Chat App",
  )
  st.header("Chat app using MCP")

  # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=TEMPERATUE)

  # ここはあとから修正
  # 性格をPOSTして切り替えられると面白そう
  if "messages" not in st.session_state:
    st.session_state.messages = [
      SystemMessage(content="あなたは優秀なAIエージェントです。気さくで楽しく明るい性格で、ユーザの入力に対してユーモラスに返答します。")
    ]
  if user_input := st.chat_input("何でも入力してね！"):
    # 会話履歴はlangchainのオブジェクトで管理する
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("AI agent is typing..."):
      # 会話履歴 + ユーザの入力をjsonにしてAPIで推論サーバにpost
      json_data = messages_to_dict(st.session_state.messages)
      json_str = json.dumps(json_data, indent=2)
      print("=================================")
      print(f"send data")
      print(json_str)
      response = requests.post(API_URL, json={"message": json_str})
      print(f"received data")
      json_data = response.json()
      json_data = json.loads(json_data["response"])
      
      print(type(json_data))
      print(json_data)
      # json_data = json.loads(response.content.decode("utf-8"))
      # print(json_data["response"])
      # response_data = response.json()
      # print("")

    # 推論サーバからデータがjsonで来るのでlangchainのオブジェクトに変換
    recevied_msgs = messages_from_dict(json_data)
    for msg in recevied_msgs:
      st.session_state.messages.append(msg)

    messages = st.session_state.get("messages", [])
    for message in messages:
      content = message.content
      if isinstance(message, AIMessage):
        with st.chat_message("AI agent"):
          st.markdown(content)
      elif isinstance(message, HumanMessage):
        with st.chat_message("User"):
          st.markdown(content)
      else:
        st.write(f"System: {content}")

if __name__ == "__main__":
  main()
