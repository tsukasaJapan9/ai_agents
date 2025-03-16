import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import GitLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def file_filter(file_path: str) -> bool:
  return file_path.endswith(".mdx")


# ドキュメントをembedding
doc_embeddings = GoogleGenerativeAIEmbeddings(
  model="models/embedding-001", task_type="retrieval_document"
)

# ユーザのクエリをembedding
query_embeddings = GoogleGenerativeAIEmbeddings(
  model="models/embedding-001", task_type="retrieval_query"
)

if not os.path.exists("./rag_db"):
  print("DB is not found. Creating DB...")

  loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
  )

  documents = loader.load()
  print(len(documents))

  db = Chroma.from_documents(documents, doc_embeddings, persist_directory="./rag_db")
else:
  print("DB is found. Loading DB...")
  db = Chroma(persist_directory="./rag_db", embedding_function=doc_embeddings)

prompt_text = """以下の文脈だけを踏まえて質問に回答してください。

文脈:
{context}

質問:
{question}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)
retriever = db.as_retriever()

# doc = db.similarity_search("langchainのAPIについて教えて")
# print(doc)

chain = (
  {
    "question": RunnablePassthrough(),
    "context": retriever,
  }
  | prompt
  | model
  | StrOutputParser()
)

result = chain.invoke("langchainとはなんですか？")
print(result)
