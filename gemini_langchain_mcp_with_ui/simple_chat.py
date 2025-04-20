import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

TEMPERATUE = 0.5

def main() -> None:
  # UIの初期化
  st.set_page_config(
    page_title="Chat App",
  )
  st.header("Self made chat app")

  llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=TEMPERATUE)

  if "messages" not in st.session_state:
    st.session_state.messages = [
      SystemMessage(content="あなたは優秀なAIエージェントです。気さくで楽しく明るい性格で、ユーザの入力に対してユーモラスに返答します。")
    ]
  if user_input := st.chat_input("何でも入力してね！"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("AI agent is typing..."):
      response = llm.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))

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
