import operator
import os
from typing import Annotated, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "agent-book"

ROLES = {
  "1": {
    "name": "一般知識エキスパート",
    "description": "幅広い分野の一般的な質問に答える",
    "details": "幅広い分野の一般的な質問に対して正確でわかりやすい回答を提供してください",
  },
  "2": {
    "name": "生成AI製品エキスパート",
    "description": "生成AIや関連製品、技術に関する専門的な質問に答える",
    "details": "生成AIや関連製品、技術に関する専門的な質問に対して、最新の情報と深い洞察を提供してください",
  },
  "3": {
    "name": "カウンセラー",
    "description": "個人的な悩みや心理的な問題に対してサポートを提供する",
    "details": "個人的な悩みや心理的な問題に対して、共感的で支援的な回答を提供し、可能であれば適切なアドバイスも行ってください",
  },
}

selection_prompt = """
質問を分析し、最も適切な回答担当ロールを選択してください。

選択肢: {role_options}

回答は選択肢の番号(1, 2, 3)のみを返してください。

質問: {query}
""".strip()

answering_prompt = """
あなたは{role}として回答してください。以下の質問に対して、
あなたの役割に基づいた適切な回答を提供してください。
回答は完結に100文字以内でお願いします。

役割の詳細:
{role_details}

質問:
{query}

回答:
""".strip()

judgement_prompt = """
以下の回答の品質をチェックし、問題がある場合は'False'、問題がない場合は'True'を返してください。
またその判断理由も50文字以内で完結に説明してください。

ユーザからの質問:
{query}

回答:
{answer}
""".strip()


class State(BaseModel):
  query: str = Field(
    ...,
    description="ユーザからの質問",
  )
  current_role: str = Field(default="", description="選定された回答ロール")
  messages: Annotated[list[str], operator.add] = Field(
    default=[],
    description="回答履歴",
  )
  current_judge: bool = Field(
    default=False,
    description="品質チェックの結果",
  )
  judgement_reason: str = Field(
    default="",
    description="品質チェックの判定理由",
  )


_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
llm = _llm.configurable_fields(max_tokens=ConfigurableField(id="max_tokens"))


def selection_node(state: State) -> dict[str, Any]:
  query = state.query
  role_options = "\n".join(
    [
      f"{role_id}: {role['name']} - {role['description']}"
      for role_id, role in ROLES.items()
    ]
  )
  prompt = ChatPromptTemplate.from_template(selection_prompt)

  chain = prompt | llm.with_config(configurable=dict(max_tokens=1)) | StrOutputParser()
  role_number: str = chain.invoke({"role_options": role_options, "query": query})
  selected_role = ROLES[role_number.strip()]["name"]
  return {"current_role": selected_role}


def answering_node(state: State) -> dict[str, Any]:
  query = state.query
  role = state.current_role
  role_details = "\n".join(
    [f"- {role['name']}: {role['details']}" for role in ROLES.values()]
  )

  prompt = ChatPromptTemplate.from_template(answering_prompt)

  chain = prompt | llm | StrOutputParser()
  answer: str = chain.invoke(
    {"role": role, "role_details": role_details, "query": query}
  )
  return {"messages": [answer]}


class Judgement(BaseModel):
  reason: str = Field(
    default="",
    description="判定理由",
  )
  judge: bool = Field(
    default=False,
    description="判定結果",
  )


def check_node(state: State) -> dict[str, Any]:
  query = state.query
  answer = state.messages[-1]
  prompt = ChatPromptTemplate.from_template(judgement_prompt)

  chain = prompt | llm.with_structured_output(Judgement)
  result: Judgement = chain.invoke({"query": query, "answer": answer})
  return {"current_judge": result.judge, "judgement_reason": result.reason}


workflow = StateGraph(State)

workflow.add_node("selection", selection_node)
workflow.add_node("answering", answering_node)
workflow.add_node("check", check_node)

workflow.set_entry_point("selection")

workflow.add_edge("selection", "answering")
workflow.add_edge("answering", "check")

workflow.add_conditional_edges(
  "check", lambda state: state.current_judge, {True: END, False: "selection"}
)

compiled = workflow.compile()

print(compiled.get_graph().draw_ascii())

# # initial_state = State(query="生成AIの最新動向について教えてください。")
# initial_state = State(query="最近疲れています、どうすればいいですか？")
# result = compiled.invoke(initial_state)

# print(result)
