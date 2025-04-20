import json
import os
from contextlib import AsyncExitStack

from fastapi import FastAPI
from langchain_core.messages import ToolMessage
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

# for langsmith
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "mcp agent"

GOOGLE_CUSTOM_SEARCH_API_KEY = os.environ["GOOGLE_CUSTOM_SEARCH_API_KEY"]
GOOGLE_CUSTOM_SEARCH_ENGINE_ID = os.environ["GOOGLE_CUSTOM_SEARCH_ENGINE_ID"]


TEMPERATUE = 0.5
# SERVER_SCRIPT = "./local_cmd_mcp_server.py"
SERVER_SCRIPT = "./custom_search_mcp_server.py"


class UserInput(BaseModel):
  message: str


app = FastAPI()

read_stream = None
write_stream = None
session = None
exit_stack = AsyncExitStack()
agent = None
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=TEMPERATUE)
tools = []
resources = []


@app.on_event("startup")
async def startup_event():
  global read_stream, write_stream, session, exit_stack, agent, tools, resources
  server_params = StdioServerParameters(
    command="python",
    args=[SERVER_SCRIPT],
    env={
      "GOOGLE_CUSTOM_SEARCH_API_KEY": GOOGLE_CUSTOM_SEARCH_API_KEY,
      "GOOGLE_CUSTOM_SEARCH_ENGINE_ID": GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
    },
  )
  await exit_stack.__aenter__()
  read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(server_params))
  session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))

  await session.initialize()

  tools = await load_mcp_tools(session)
  print("--------------------------")
  print("Available tools")
  print("--------------------------")
  for tool in tools:
    print(f"{tool.name}: {tool.description}")

  resources = await load_mcp_resources(session)
  print("--------------------------")
  print("Available resources")
  print("--------------------------")
  for resource in resources:
    print(f"{resource.data}, {resource.mimetype}, {resource.metadata}")

  agent = create_react_agent(model, tools)


@app.on_event("shutdown")
async def shutdown():
  global exit_stack
  await exit_stack.__aexit__(None, None, None)


@app.post("/infer")
async def infer(input_data: UserInput):
  global session, agent

  print("=============================================")
  # ユーザからの入力はAPI経由でjsonでくるのでlangchainのメッセージオブジェクトに変換
  messages = messages_from_dict(json.loads(input_data.message))

  print("---------- [infer server]: user input data from ui ----------")
  for message in messages:
    print(f"{message.__class__.__name__}: {message.content}")

  agent_response = await agent.ainvoke({"messages": messages})
  agent_response = agent_response["messages"]

  print("---------- [infer server]: ai agent response ----------")

  for message in agent_response:
    if isinstance(message, ToolMessage):
      message.content = json.loads(message.content)
    print(f"{message.__class__.__name__}: {message.content}, {type(message)}")

  # langchainのinvokeで得たデータはメッセージオブジェクトなのでjsonに変換
  json_data = messages_to_dict(agent_response)
  json_str = json.dumps(json_data, indent=2)
  return {"response": json_str}


@app.get("/tools")
def get_tools() -> dict[str, str]:
  _tools = {tool.name: tool.description for tool in tools}
  return {"tools": json.dumps(_tools, indent=2)}
