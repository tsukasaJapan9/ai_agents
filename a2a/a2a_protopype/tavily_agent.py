import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# .envから環境変数を読み込む
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


@dataclass
class AgentCard:
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    provider: Optional[Dict[str, str]] = None
    authentication: Optional[Dict[str, Any]] = None


# Web検索ツールの初期化
search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# --- A2Aエンドポイント実装 ---


async def agent_card(request):
    """/.well-known/agent.json: エージェントカードを返す"""
    card = AgentCard(
        name="Tavily Web Search Agent",
        description="Tavily APIを使ったWeb検索エージェント。クエリに対して最新のWeb検索結果を返します。",
        url=str(request.url.with_path("").with_query("").with_fragment("")),
        skills=[{"name": "web_search", "description": "Web検索"}],
        capabilities={"search": True},
        provider={"name": "Tavily"},
        authentication={"type": "apiKey"},
    )
    return web.json_response(card.__dict__)


async def a2a_endpoint(request):
    """/a2a: JSON-RPC 2.0エンドポイント"""
    req = await request.json()
    method = req.get("method")
    params = req.get("params", {})
    id_ = req.get("id", str(uuid.uuid4()))

    if method == "tasks/send":
        # ユーザーからの検索クエリを受け取る
        message = params.get("message", {})
        query = ""
        if message and isinstance(message.get("parts"), list):
            for part in message["parts"]:
                if part.get("type") == "text":
                    query = part.get("text", "")
                    break
        if not query:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": id_,
                    "error": {"code": -32602, "message": "No query provided."},
                }
            )
        # Tavilyで検索
        try:
            results = search_tool.invoke({"query": query})
            # 結果を文字列として整形
            if isinstance(results, list):
                formatted_results = []
                for result in results:
                    title = result.get("title", "")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    formatted_results.append(
                        f"タイトル: {title}\n内容: {content}\nURL: {url}\n"
                    )
                results_text = "\n".join(formatted_results)
            else:
                results_text = str(results)

            # 結果をA2AのTask形式で返す
            task = {
                "id": params.get("id", str(uuid.uuid4())),
                "status": {"state": "completed"},
                "message_history": [
                    {"role": "user", "parts": [{"type": "text", "text": query}]},
                    {
                        "role": "agent",
                        "parts": [{"type": "text", "text": results_text}],
                    },
                ],
            }
            return web.json_response({"jsonrpc": "2.0", "id": id_, "result": task})
        except Exception as e:
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": id_,
                    "error": {"code": -32000, "message": str(e)},
                }
            )
    elif method == "tasks/get":
        # 単純な実装: すべて即時完了として返す
        task_id = params.get("id")
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "id": id_,
                "result": {"id": task_id, "status": {"state": "completed"}},
            }
        )
    else:
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "id": id_,
                "error": {"code": -32601, "message": "Method not found."},
            }
        )


# --- aiohttpアプリケーション ---


def create_app():
    app = web.Application()
    app.router.add_get("/.well-known/agent.json", agent_card)
    app.router.add_post("/a2a", a2a_endpoint)
    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    web.run_app(create_app(), port=port)
