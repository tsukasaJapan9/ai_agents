import operator
import os
from typing import Annotated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "agent-book"


user_request = (
  "新しいコンシューマ向けプロダクトの開発に関するアイデアを提案してください。"
)

persona_generator_prompt = """以下のユーザーリクエストに関するインタビュー用に、
{k}人の多様なペルソナを生成してください。

ユーザーリクエスト: {user_request}

各ペルソナには名前と簡単な背景を含めてください。
年齢、性別、職業、技術的専門知識において多様性を持たせるようにしてください。
また、背景は100文字以内で簡潔に記述してください。
"""

interview_conductor_question_prompt = """以下のペルソナに関連するユーザーリクエストに対して、1つの質問を生成してください。

ユーザリクエスト: {user_request}
ペルソナ: {persona_name} - {persona_background}

質問は具体的で、このペルソナの視点から重要な情報を引き出すように質問を設計してください。
質問は100文字以内で簡潔に記述してください。
"""

evaludator_prompt = """以下のユーザーリクエストとインタビュー結果に基づいて、
包括的な要件文書を作成するための情報が十分に集まったかどうかを判断してください

ユーザーリクエスト: {user_request}
インタビュー結果: {interviews}
"""

req_doc_prompt = """以下のユーザリクエストと複数のペルソナからのインタビュー結果に基づき
要件文書を作成してください。

ユーザリクエスト: {user_request}
インタビュー結果: {interviews}
要件文書には以下のセクションを含めてください。
1. プロジェクト概要
2. 主要機能
3. 非機能要件
4. 制約条件
5. ターゲットユーザー
6. マネタイズの方法
7. 考えうるリスクと軽減策

出力は日本語でお願いします。
"""


class Persona(BaseModel):
  name: str = Field(
    ...,
    description="ペルソナの名前",
  )
  background: str = Field(
    ...,
    description="ペルソナの背景",
  )


class Personas(BaseModel):
  personas: list[Persona] = Field(default_factory=list, description="ペルソナのリスト")


class Interview(BaseModel):
  persona: Persona = Field(..., description="インタビュー対象のペルソナ")
  question: str = Field(..., description="インタビューでの質問")
  answer: str = Field(..., description="インタビューでの回答")


class Interviews(BaseModel):
  interviews: list[Interview] = Field(
    default_factory=list, description="インタビュー結果のリスト"
  )


class InterviewResult(BaseModel):
  interviews: list[Interview] = Field(
    default_factory=list, description="インタビュー結果のリスト"
  )


class EvaluationResult(BaseModel):
  reason: str = Field(..., description="判断の理由")
  is_sufficient: bool = Field(..., description="情報が十分かどうか")


class InteviewState(BaseModel):
  user_request: str = Field(..., description="ユーザーからのリクエスト")
  personas: Annotated[list[Interview], operator.add] = Field(
    default_factory=list, description="生成されたペルソナのリスト"
  )

  interviews: Annotated[list[Interview], operator.add] = Field(
    default_factory=list, description="実施されたインタビューのリスト"
  )

  requirements_doc: str = Field(
    default="",
    description="生成された要件定義",
  )

  iteration: int = Field(
    default=0,
    description="ペルソナ生成とインタビューの反復回数",
  )

  is_infomation_sufficient: bool = Field(
    default=False,
    description="情報が十分かどうか",
  )


class PersonaGenerator:
  def __init__(self, llm: ChatOpenAI, k: int = 3):
    self.llm = llm.with_structured_output(Personas)
    self.k = k

  def run(self, user_request: str) -> Personas:
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "あなたはユーザーインタビュー用の多様なペルソナを生成する専門家です",
        ),
        (
          "human",
          persona_generator_prompt.format(user_request=user_request, k=self.k),
        ),
      ]
    )
    chain = prompt | self.llm
    return chain.invoke({"user_request": user_request})


class InterviewConductor:
  def __init__(self, llm: ChatOpenAI):
    self.llm = llm

  def _generate_questions(
    self, user_request: str, personas: list[Persona]
  ) -> list[str]:
    question_prompts = ChatPromptTemplate.from_messages(
      [
        ("system", "あなたはユーザー要件に基づいて適切な質問を生成する専門家です"),
        ("human", interview_conductor_question_prompt),
      ]
    )

    print(f"{question_prompts=}")
    chain = question_prompts | self.llm | StrOutputParser()
    question_queries = [
      {
        "user_request": user_request,
        "persona_name": persona.name,
        "persona_background": persona.background,
      }
      for persona in personas
    ]
    print(f"{question_queries=}")
    return chain.batch(question_queries)

  def _generate_answers(
    self, personas: list[Persona], questions: list[str]
  ) -> list[str]:
    answer_prompts = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "あなたは以下のペルソナとして回答しています: {persona_name} - [persona_background]",
        ),
        ("human", "質問: {question}"),
      ]
    )
    print(f"{answer_prompts=}")
    chain = answer_prompts | self.llm | StrOutputParser()
    answer_queries = [
      {
        "persona_name": persona.name,
        "persona_background": persona.background,
        "question": question + "100文字以内で回答してください。",
      }
      for persona, question in zip(personas, questions)
    ]
    print(f"{answer_queries=}")
    return chain.batch(answer_queries)

  def _create_interviews(
    self, personas: list[Persona], questions: list[str], answers: list[str]
  ) -> list[Interview]:
    return [
      Interview(persona=persona, question=question, answer=answer)
      for persona, question, answer in zip(personas, questions, answers)
    ]

  def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
    questions = self._generate_questions(user_request=user_request, personas=personas)
    answers = self._generate_answers(personas=personas, questions=questions)

    interviews = self._create_interviews(
      personas=personas, questions=questions, answers=answers
    )

    return InterviewResult(interviews=interviews)


class InformationEvaluator:
  def __init__(self, llm: ChatOpenAI):
    self.llm = llm.with_structured_output(EvaluationResult)

  def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です",
        ),
        (
          "human",
          evaludator_prompt.format(user_request=user_request, interviews=interviews),
        ),
      ]
    )
    interview_results = "\n".join(
      [
        f"ペルソナ: {i.persona.name} - {i.persona.background}\n質問: {i.question}\n回答: {i.answer}"
        for i in interviews
      ]
    )
    chain = prompt | self.llm
    return chain.invoke({"user_request": user_request, "interviews": interview_results})


class RequirementDocumentGenerator:
  def __init__(self, llm: ChatOpenAI):
    self.llm = llm

  def run(self, user_request: str, interviews: list[Interview]) -> str:
    prompt = ChatPromptTemplate.from_messages(
      [
        (
          "system",
          "あなたは収集した情報に基づき要件文書を作成する専門家です",
        ),
        (
          "human",
          req_doc_prompt.format(user_request=user_request, interviews=interviews),
        ),
      ]
    )
    interview_results = "\n".join(
      [
        f"ペルソナ: {i.persona.name} - {i.persona.background}\n質問: {i.question}\n回答: {i.answer}"
        for i in interviews
      ]
    )
    chain = prompt | self.llm | StrOutputParser()
    return chain.invoke({"user_request": user_request, "interviews": interview_results})


ChatOpenAI(model="gpt-4o", temperature=0.0)
_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
persona_generator = PersonaGenerator(_llm)
personas = persona_generator.run(user_request=user_request)

interview_conductor = InterviewConductor(_llm)
interviews = interview_conductor.run(
  user_request=user_request,
  personas=personas.personas,
)

print(interviews)

evaluator = InformationEvaluator(_llm)
evaluation_result = evaluator.run(
  user_request=user_request,
  interviews=interviews.interviews,
)

print(evaluation_result)

req_doc_generator = RequirementDocumentGenerator(_llm)
req_doc = req_doc_generator.run(
  user_request=user_request, interviews=interviews.interviews
)

print(req_doc)
