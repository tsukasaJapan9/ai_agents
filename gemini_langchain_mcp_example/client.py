# このファイルで実装すること
# - langchainを通してgemini 2.0 flashを使う
# - ユーザの入力にしたがってLLMから解答を得る
# - MCPを使ってLLMがToolを使えるようにする

import asyncio
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

TEMPERATUE = 0.5


async def run(server_script: str) -> None:
  model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=TEMPERATUE)
  server_params = StdioServerParameters(command="python", args=[server_script])
  async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
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

      query_history = []

      while True:
        query = input("\nQuery: ").strip()
        if query.lower() == "exit":
          break
        query_history.append(query)
        agent_response = await agent.ainvoke({"messages": query_history})
        query_history = []
        print("-----------")
        for message in agent_response["messages"]:
          print(f"{message.__class__.__name__}: {message.content}")
          query_history.append(message)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("args error")
    sys.exit(1)

  server_script = sys.argv[1]
  asyncio.run(run(server_script))
