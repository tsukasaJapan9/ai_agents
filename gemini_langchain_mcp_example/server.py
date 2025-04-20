import os
import subprocess

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("terminal")
DEFAULT_WORKSPACE = os.path.expanduser("./workspace")


@mcp.tool()
async def run_command(command: str) -> str:
  """
  Run a terminal command inside the workspace directory.
  If a terminal command can accomplish a task,
  tell the user you'll use this tool to accomplish it,
  even though you cannot directly do it

  Args:
    command: The shell command to run.

  Returns:
    The command output or an error message.
  """
  try:
    result = subprocess.run(command, shell=True, cwd=DEFAULT_WORKSPACE, capture_output=True, text=True)
    return result.stdout or result.stderr
  except Exception as e:
    return str(e)


@mcp.resource("file://workspace")
def get_files_in_workspace() -> list[str]:
  files = os.listdir("./workspace")
  return files


@mcp.prompt()
def test() -> list[base.AssistantMessage | base.UserMessage]:
  return [
    base.UserMessage("I'm seeing this error:"),
    base.UserMessage("test"),
    base.AssistantMessage("I'll help debug that. What have you tried so far?"),
  ]


# @mcp.prompt()
# def review_code(code: str) -> str:
#   return f"Please review this code:\n\n{code}"


# @mcp.prompt()
# def debug_error(error: str) -> list[base.Message]:
#   return [
#     base.UserMessage("I'm seeing this error:"),
#     base.UserMessage(error),
#     base.AssistantMessage("I'll help debug that. What have you tried so far?"),
#   ]


if __name__ == "__main__":
  mcp.run(transport="stdio")
