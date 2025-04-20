import json
from contextlib import AsyncExitStack

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

TEMPERATUE = 0.5
SERVER_SCRIPT = "./server.py"

class UserInput(BaseModel):
    message: str

app = FastAPI()

read_stream = None
write_stream = None
session = None
exit_stack = AsyncExitStack()
agent = None
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=TEMPERATUE)

@app.on_event("startup")
async def startup_event():
  global read_stream, write_stream, session, exit_stack, agent
  server_params = StdioServerParameters(command="python", args=[SERVER_SCRIPT])
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

  print(f"query message: {input_data.message}")
  print(f"query message: {type(input_data.message)}")

  # ユーザからの入力はAPI経由でjsonでくるのでlangchainのメッセージオブジェクトに変換
  queries = messages_from_dict(json.loads(input_data.message))
  agent_response = await agent.ainvoke({"messages": queries})
  agent_response = agent_response["messages"]

  print("AI agent response")

  for message in agent_response:
    print(f"{message.__class__.__name__}: {message.content}")

  # langchainのinvokeで得たデータはメッセージオブジェクトなのでjsonに変換
  json_data = messages_to_dict(agent_response)
  json_str = json.dumps(json_data, indent=2)
  return {"response": json_str}
