import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urljoin

import nest_asyncio
import uvicorn
from aiohttp import ClientSession, ClientTimeout
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

# .envファイルから環境変数を読み込む
load_dotenv()

# イベントループの問題を解決
nest_asyncio.apply()

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentCard:
    """エージェントの能力を記述するカード"""

    name: str
    description: Optional[str] = None
    url: str = ""
    version: str = "1.0.0"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    provider: Optional[Dict[str, str]] = None
    authentication: Optional[Dict[str, Any]] = None


@dataclass
class Message:
    """A2Aプロトコルのメッセージ"""

    role: str
    parts: List[Dict[str, Any]]


@dataclass
class Task:
    """A2Aプロトコルのタスク"""

    id: str
    status: Dict[str, str]
    message_history: Optional[List[Message]] = None


@dataclass
class TaskSendParams:
    """タスク送信パラメータ"""

    id: str
    message: Message
    session_id: Optional[str] = None
    accepted_output_modes: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentState(TypedDict):
    """LangGraphエージェントの状態"""

    messages: List[Any]
    available_agents: List[Dict[str, Any]]
    current_task: Optional[Dict[str, Any]]
    task_results: List[Dict[str, Any]]


class A2AClient:
    """A2Aプロトコルクライアント"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = ClientTimeout(total=timeout)
        self.session: Optional[ClientSession] = None

    async def __aenter__(self):
        self.session = ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        if self.session:
            await self.session.close()

    async def get_agent_card(self) -> Optional[AgentCard]:
        """エージェントカードを取得"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        try:
            url = urljoin(self.base_url, "/.well-known/agent.json")
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return AgentCard(**data)
                else:
                    logger.warning(f"Failed to get agent card: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting agent card: {e}")
            return None

    async def send_task(self, params: TaskSendParams) -> Optional[Task]:
        """タスクを送信"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_data: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": params.id,
                "message": {"role": params.message.role, "parts": params.message.parts},
            },
        }

        if params.session_id:
            request_data["params"]["sessionId"] = params.session_id
        if params.accepted_output_modes:
            request_data["params"]["acceptedOutputModes"] = params.accepted_output_modes
        if params.metadata:
            request_data["params"]["metadata"] = params.metadata

        try:
            url = urljoin(self.base_url, "/a2a")
            async with self.session.post(url, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        return Task(**data["result"])
                    elif "error" in data:
                        logger.error(f"A2A error: {data['error']}")
                        return None
                else:
                    logger.error(f"HTTP error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error sending task: {e}")
            return None

    async def get_task(self, task_id: str) -> Optional[Task]:
        """タスクの状態を取得"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id},
        }

        try:
            url = urljoin(self.base_url, "/a2a")
            async with self.session.post(url, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        result = data["result"]
                        # message_historyが含まれていない場合は、空のリストを設定
                        if "message_history" not in result:
                            result["message_history"] = []
                        return Task(**result)
                    elif "error" in data:
                        logger.error(f"A2A error: {data['error']}")
                        return None
                else:
                    logger.error(f"HTTP error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting task: {e}")
            return None

    async def cancel_task(self, task_id: str) -> bool:
        """タスクをキャンセル"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/cancel",
            "params": {"id": task_id},
        }

        try:
            url = urljoin(self.base_url, "/a2a")
            async with self.session.post(url, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data:
                        return True
                    elif "error" in data:
                        logger.error(f"A2A error: {data['error']}")
                        return False
                else:
                    logger.error(f"HTTP error: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error canceling task: {e}")
            return False

    async def list_tasks(self) -> List[str]:
        """アクティブなタスクの一覧を取得"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        request_data = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/list",
            "params": {},
        }

        try:
            url = urljoin(self.base_url, "/a2a")
            async with self.session.post(url, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    if "result" in data and "tasks" in data["result"]:
                        return [task["id"] for task in data["result"]["tasks"]]
                    elif "error" in data:
                        logger.error(f"A2A error: {data['error']}")
                        return []
                else:
                    logger.error(f"HTTP error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []


class MultiAgentOrchestrator:
    """マルチエージェントオーケストレータ（LangGraph + A2A）"""

    def __init__(self, name: str = "Multi-Agent Orchestrator"):
        self.name = name
        self.registered_agents: Dict[str, AgentCard] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            max_tokens=2000,
        )

        # LangGraphエージェントとツールを初期化
        self.tools = self._create_tools()
        self.agent = create_react_agent(self.llm, self.tools)
        self.workflow = self._create_workflow()

    def _create_tools(self) -> List[Any]:
        """A2Aエージェント操作用のツールを作成"""

        @tool
        def discover_agent(agent_url: str) -> str:
            """エージェントを発見して登録する"""
            try:
                # 非同期関数を同期的に実行
                loop = asyncio.get_event_loop()
                agent_card = loop.run_until_complete(
                    self._discover_agent_async(agent_url)
                )
                if agent_card:
                    self.registered_agents[agent_card.name] = agent_card
                    return f"エージェント '{agent_card.name}' を発見して登録しました。"
                else:
                    return f"エージェントの発見に失敗しました: {agent_url}"
            except Exception as e:
                return f"エージェント発見エラー: {str(e)}"

        @tool
        def list_available_agents() -> str:
            """利用可能なエージェントの一覧を取得する"""
            if not self.registered_agents:
                return "登録されたエージェントがありません。"

            agent_list = []
            for agent_id, agent in self.registered_agents.items():
                agent_info = {
                    "id": agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "skills": agent.skills,
                }
                agent_list.append(agent_info)

            return json.dumps(agent_list, ensure_ascii=False, indent=2)

        @tool
        def send_task_to_agent(agent_name: str, task_message: str) -> str:
            """指定されたエージェントにタスクを送信する"""
            try:
                if agent_name not in self.registered_agents:
                    return f"エージェント '{agent_name}' が見つかりません。"

                agent = self.registered_agents[agent_name]
                task_id = str(uuid.uuid4())

                # 非同期関数を同期的に実行
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    self._send_task_async(agent.url, task_id, task_message)
                )

                if result:
                    self.active_tasks[task_id] = result
                    return f"タスク '{task_id}' をエージェント '{agent_name}' に送信しました。"
                else:
                    return "タスクの送信に失敗しました。"
            except Exception as e:
                return f"タスク送信エラー: {str(e)}"

        @tool
        def get_task_result(task_id: str) -> str:
            """タスクの実行結果を取得する"""
            try:
                if task_id not in self.active_tasks:
                    return f"タスク '{task_id}' が見つかりません。"

                task = self.active_tasks[task_id]

                # タスクの状態を確認
                if task.status.get("state") == "completed":
                    # 完了している場合は結果を返す
                    if task.message_history and len(task.message_history) > 1:
                        # エージェントの応答（最後のメッセージ）を取得
                        agent_response = task.message_history[-1]
                        # message_historyは辞書のリストなので、辞書として扱う
                        if (
                            isinstance(agent_response, dict)
                            and agent_response.get("role") == "agent"
                            and agent_response.get("parts")
                        ):
                            result_text = ""
                            for part in agent_response["parts"]:
                                if part.get("type") == "text":
                                    result_text += part.get("text", "")
                            return f"タスク '{task_id}' の実行結果:\n{result_text}"
                        else:
                            return (
                                f"タスク '{task_id}' の実行結果が取得できませんでした。"
                            )
                    else:
                        return f"タスク '{task_id}' のメッセージ履歴が不足しています。"
                else:
                    # まだ実行中の場合は状態を返す
                    return f"タスク '{task_id}' はまだ実行中です。状態: {task.status.get('state', 'unknown')}"
            except Exception as e:
                return f"タスク結果取得エラー: {str(e)}"

        @tool
        def wait_for_task_completion(task_id: str, max_wait_seconds: int = 30) -> str:
            """タスクの完了を待機して結果を取得する"""
            try:
                if task_id not in self.active_tasks:
                    return f"タスク '{task_id}' が見つかりません。"

                import time

                start_time = time.time()

                while time.time() - start_time < max_wait_seconds:
                    task = self.active_tasks[task_id]
                    if task.status.get("state") == "completed":
                        # 完了したら結果を返す
                        if task.message_history and len(task.message_history) > 1:
                            agent_response = task.message_history[-1]
                            if (
                                isinstance(agent_response, dict)
                                and agent_response.get("role") == "agent"
                                and agent_response.get("parts")
                            ):
                                result_text = ""
                                for part in agent_response["parts"]:
                                    if part.get("type") == "text":
                                        result_text += part.get("text", "")
                                return f"タスク '{task_id}' の実行結果:\n{result_text}"

                    # 少し待機してから再チェック
                    time.sleep(1)

                return f"タスク '{task_id}' の完了を待機中ですが、{max_wait_seconds}秒以内に完了しませんでした。"
            except Exception as e:
                return f"タスク完了待機エラー: {str(e)}"

        @tool
        def get_task_status(task_id: str) -> str:
            """タスクの状態を取得する"""
            try:
                if task_id not in self.active_tasks:
                    return f"タスク '{task_id}' が見つかりません。"

                task = self.active_tasks[task_id]
                return json.dumps(
                    {
                        "task_id": task.id,
                        "status": task.status,
                        "message_history": [
                            {"role": msg.role, "parts": msg.parts}
                            for msg in (task.message_history or [])
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as e:
                return f"タスク状態取得エラー: {str(e)}"

        @tool
        def select_best_agent_for_task(task_description: str) -> str:
            """タスクに最適なエージェントを選択する"""
            if not self.registered_agents:
                return "利用可能なエージェントがありません。"

            # 簡単なキーワードマッチング
            task_lower = task_description.lower()
            best_agent = None
            best_score = 0

            for agent_name, agent in self.registered_agents.items():
                score = 0
                if agent.description:
                    desc_lower = agent.description.lower()
                    # キーワードマッチング
                    keywords = {
                        "weather": ["weather", "天気", "気温", "気候"],
                        "travel": ["travel", "旅行", "予約", "booking", "hotel"],
                        "code": ["code", "programming", "開発", "コード"],
                        "analysis": ["analysis", "分析", "データ", "analytics"],
                    }

                    for category, words in keywords.items():
                        if any(word in desc_lower for word in words):
                            if any(word in task_lower for word in words):
                                score += 2
                            else:
                                score += 1

                if score > best_score:
                    best_score = score
                    best_agent = agent_name

            if best_agent:
                return f"タスク '{task_description}' に最適なエージェント: {best_agent}"
            else:
                return f"タスク '{task_description}' に適したエージェントが見つかりません。最初のエージェントを使用します: {list(self.registered_agents.keys())[0]}"

        return [
            discover_agent,
            list_available_agents,
            send_task_to_agent,
            get_task_result,
            wait_for_task_completion,
            get_task_status,
            select_best_agent_for_task,
        ]

    def _create_workflow(self) -> Any:
        """LangGraphワークフローを作成"""

        def should_continue(state: AgentState) -> str:
            """次のステップを決定する"""
            messages = state["messages"]
            last_message = messages[-1]

            # 最後のメッセージがツール呼び出しの結果の場合、エージェントに続行させる
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "agent"

            # それ以外の場合は終了
            return END

        # ワークフローを作成
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.agent)
        workflow.add_conditional_edges("agent", should_continue)
        workflow.set_entry_point("agent")

        return workflow.compile()

    async def _discover_agent_async(self, agent_url: str) -> Optional[AgentCard]:
        """エージェントを非同期で発見"""
        async with A2AClient(agent_url) as client:
            return await client.get_agent_card()

    async def _send_task_async(
        self, agent_url: str, task_id: str, message_text: str
    ) -> Optional[Task]:
        """タスクを非同期で送信"""
        async with A2AClient(agent_url) as client:
            message = Message(
                role="user", parts=[{"type": "text", "text": message_text}]
            )
            task_params = TaskSendParams(
                id=task_id, message=message, accepted_output_modes=["text"]
            )
            return await client.send_task(task_params)

    async def cleanup_all_tasks(self):
        """すべてのエージェントの既存タスクをキャンセル"""
        logger.info("既存のタスクをクリーンアップ中...")
        for agent_name, agent_card in self.registered_agents.items():
            try:
                async with A2AClient(agent_card.url) as client:
                    # アクティブなタスクの一覧を取得
                    task_ids = await client.list_tasks()
                    if task_ids:
                        logger.info(
                            f"エージェント '{agent_name}' の {len(task_ids)} 個のタスクをキャンセル中..."
                        )
                        for task_id in task_ids:
                            success = await client.cancel_task(task_id)
                            if success:
                                logger.info(f"タスク '{task_id}' をキャンセルしました")
                            else:
                                logger.warning(
                                    f"タスク '{task_id}' のキャンセルに失敗しました"
                                )
                    else:
                        logger.info(
                            f"エージェント '{agent_name}' にアクティブなタスクはありません"
                        )
            except Exception as e:
                logger.error(
                    f"エージェント '{agent_name}' のタスククリーンアップ中にエラー: {e}"
                )

        # ローカルのアクティブタスクもクリア
        self.active_tasks.clear()
        logger.info("タスククリーンアップ完了")

    def register_agent(self, agent_id: str, agent_card: AgentCard):
        """エージェントを登録"""
        self.registered_agents[agent_id] = agent_card
        logger.info(f"Registered agent: {agent_card.name} ({agent_id})")

    def get_registered_agents(self) -> List[Dict[str, Any]]:
        """登録されたエージェントの一覧を取得"""
        return [
            {
                "id": agent_id,
                "name": card.name,
                "description": card.description,
                "url": card.url,
                "skills": card.skills,
            }
            for agent_id, card in self.registered_agents.items()
        ]

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """アクティブなタスクの一覧を取得"""
        return [
            {
                "id": task.id,
                "status": task.status,
                "message_history": task.message_history,
            }
            for task in self.active_tasks.values()
        ]

    async def process_user_request(self, user_message: str) -> str:
        """ユーザーリクエストを処理（LangGraphワークフローを使用）"""
        try:
            # 初期状態を作成
            initial_state = AgentState(
                messages=[HumanMessage(content=user_message)],
                available_agents=self.get_registered_agents(),
                current_task=None,
                task_results=[],
            )

            # ワークフローを実行
            result = await self.workflow.ainvoke(initial_state)

            # 結果から応答を抽出
            messages = result["messages"]
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                else:
                    return "応答を生成できませんでした。"
            else:
                return "応答が生成されませんでした。"

        except Exception as e:
            logger.error(f"Error in process_user_request: {e}")
            return f"エラーが発生しました: {str(e)}"

    async def orchestrate_complex_task(
        self, task_description: str, agent_urls: List[str]
    ) -> str:
        """複雑なタスクを複数のエージェントで協調実行"""
        try:
            # 1. エージェントを発見・登録
            for url in agent_urls:
                agent_card = await self._discover_agent_async(url)
                if agent_card:
                    self.register_agent(agent_card.name, agent_card)

            # 2. 最適なエージェントを選択してタスクを実行
            agent_selection_prompt = f"""
            以下のタスクを実行するために、利用可能なエージェントから最適なものを選択してください：
            
            タスク: {task_description}
            
            利用可能なエージェント:
            {json.dumps(self.get_registered_agents(), ensure_ascii=False, indent=2)}
            
            手順：
            1. 最適なエージェントを選択してください
            2. 選択したエージェントにタスクを送信してください
            3. タスクの完了を待機して結果を取得してください
            4. 取得した結果を返してください
            
            必ず結果を取得してから応答を完了してください。
            """

            # 3. LangGraphワークフローで処理
            result = await self.process_user_request(agent_selection_prompt)

            return result

        except Exception as e:
            logger.error(f"Error in orchestrate_complex_task: {e}")
            return f"複雑なタスクの実行中にエラーが発生しました: {str(e)}"


# グローバル変数としてオーケストレータを保持
orchestrator: Optional[MultiAgentOrchestrator] = None

# FastAPIアプリケーションを作成
app = FastAPI(title="Multi-Agent Host", version="1.0.0")

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_request(request: Dict[str, str]):
    """ユーザーリクエストを処理するHTTPエンドポイント"""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    user_input = request.get("user_input", "")
    if not user_input:
        raise HTTPException(status_code=400, detail="user_input is required")

    try:
        result = await orchestrator.process_user_request(user_input)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def get_agents():
    """登録されたエージェントの一覧を取得"""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    return {"agents": orchestrator.get_registered_agents()}


@app.get("/tasks")
async def get_tasks():
    """アクティブなタスクの一覧を取得"""
    global orchestrator
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    return {"tasks": orchestrator.get_active_tasks()}


# 使用例
async def main():
    """使用例"""
    global orchestrator

    # Google API キーが設定されているかチェック
    if not os.getenv("GOOGLE_API_KEY"):
        print("警告: GOOGLE_API_KEY環境変数が設定されていません。")
        print("LangGraph機能を使用するには、GOOGLE_API_KEYを設定してください。")
        return

    # マルチエージェントオーケストレータを作成
    orchestrator = MultiAgentOrchestrator("My Multi-Agent Orchestrator")

    # エージェントを発見して登録
    agent_urls = [
        "http://localhost:8000",  # 天気エージェント
        "http://localhost:8001",  # 旅行エージェント
        "http://localhost:8002",  # コードエージェント
        "http://localhost:8003",  # Tavily Web検索エージェント
    ]

    for url in agent_urls:
        agent_card = await orchestrator._discover_agent_async(url)
        if agent_card:
            orchestrator.register_agent(agent_card.name, agent_card)
            print(f"エージェントを登録しました: {agent_card.name}")

    # 既存のタスクをクリーンアップ
    await orchestrator.cleanup_all_tasks()

    # 登録されたエージェントの一覧を表示
    agents = orchestrator.get_registered_agents()
    print(f"登録されたエージェント: {len(agents)}個")

    # コマンドライン引数をチェック
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # FastAPIサーバーモード
        print("FastAPIサーバーモードで起動します...")
        print("HTTPエンドポイント: http://localhost:8001")
        print("ターミナルインターフェース: http://localhost:8001/docs")
        uvicorn.run(app, host="0.0.0.0", port=8001)
    else:
        # ターミナルインターフェースモード
        print("\n=== マルチエージェントシステム ===")
        print("何でも聞いてください。'quit' または 'exit' で終了します。")
        print("'server' と入力するとFastAPIサーバーモードに切り替わります。")
        print("'help' で利用可能なコマンドを表示します。")
        print("=" * 50)

        while True:
            try:
                # ユーザー入力を受け取る
                user_input = input("\nあなた: ").strip()

                # 終了コマンドのチェック
                if user_input.lower() in ["quit", "exit", "終了"]:
                    print("システムを終了します。")
                    break

                # サーバーモード切り替えコマンドのチェック
                if user_input.lower() == "server":
                    print("FastAPIサーバーモードに切り替えます...")
                    print("HTTPエンドポイント: http://localhost:8001")
                    print("ターミナルインターフェース: http://localhost:8001/docs")
                    uvicorn.run(app, host="0.0.0.0", port=8001)
                    break

                # エージェント一覧表示コマンド
                if user_input.lower() == "alist":
                    agents = orchestrator.get_registered_agents()
                    print(f"\n=== 登録されたエージェント ({len(agents)}個) ===")
                    for i, agent in enumerate(agents, 1):
                        print(f"{i}. {agent['name']}")
                        print(f"   説明: {agent['description']}")
                        print(f"   URL: {agent['url']}")
                        if agent["skills"]:
                            skills = [
                                skill.get("name", "") for skill in agent["skills"]
                            ]
                            print(f"   スキル: {', '.join(skills)}")
                        print()
                    continue

                # タスク一覧表示コマンド
                if user_input.lower() == "tlist":
                    tasks = orchestrator.get_active_tasks()
                    print(f"\n=== アクティブなタスク ({len(tasks)}個) ===")
                    if tasks:
                        for i, task in enumerate(tasks, 1):
                            print(f"{i}. タスクID: {task['id']}")
                            print(f"   状態: {task['status'].get('state', 'unknown')}")
                            if task["message_history"]:
                                print(
                                    f"   メッセージ数: {len(task['message_history'])}"
                                )
                            print()
                    else:
                        print("アクティブなタスクはありません。")
                    continue

                # ヘルプコマンド
                if user_input.lower() in ["help", "h", "?"]:
                    print("\n=== 利用可能なコマンド ===")
                    print("alist    - 登録されたエージェントの一覧を表示")
                    print("tlist    - アクティブなタスクの一覧を表示")
                    print("server   - FastAPIサーバーモードに切り替え")
                    print("help/h/? - このヘルプを表示")
                    print("quit/exit/終了 - システムを終了")
                    print("その他   - 質問やタスクを実行")
                    print("=" * 30)
                    continue

                # 空の入力をスキップ
                if not user_input:
                    continue

                print("処理中...")

                # ユーザーリクエストを処理
                result = await orchestrator.process_user_request(user_input)

                print(f"\nシステム: {result}")

                # アクティブなタスクの一覧を表示
                tasks = orchestrator.get_active_tasks()
                if tasks:
                    print(f"\nアクティブなタスク: {len(tasks)}個")

            except KeyboardInterrupt:
                print("\n\nシステムを終了します。")
                break
            except Exception as e:
                print(f"\nエラーが発生しました: {str(e)}")
                continue


if __name__ == "__main__":
    # イベントループの問題を回避
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            print(
                "イベントループの問題が発生しました。nest_asyncioを使用して再試行します。"
            )
            # 既存のイベントループを使用
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
        else:
            raise
