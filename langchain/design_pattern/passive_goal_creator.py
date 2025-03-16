import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "agent-book"


class Goal(BaseModel):
  description: str = Field(
    ...,
    description="目標の説明",
  )

  @property
  def text(self) -> str:
    return f"{self.description}"


prompt_text = """ユーザの入力を分析し、明確で実行可能な目標を生成してください。
要件:
1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されていること
2. あなたが実行可能な公道は以下の行動だけです。
  - インターネットを利用して、目標を達成するための調査を行う
  - ユーザのためのレポートを生成する
3. 決して2以外の行動を取ってはいけません。

ユーザの入力: {user_input}
"""

user_input = "今日は雨なので家族で楽しめるところに行きたい"

# llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "あなたはユーザの入力から目標を抽出し詳細化するための専門家です。",
    ),
    (
      "human",
      user_input,
    ),
  ]
)
chain = prompt | llm.with_structured_output(Goal)
result = chain.invoke({"user_input": user_input})

print(type(result))
print(result.text)
