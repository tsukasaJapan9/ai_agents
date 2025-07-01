# A2A Multi-Agent Orchestrator (LangGraph + Gemini 2.5 Flash)

A2A（Agent2Agent）プロトコルとLangGraphを使用したマルチエージェントオーケストレータの実装です。Google Gemini 2.5 Flashを使用して高度なタスク処理とエージェント間の協調を実現します。

## 概要

このマルチエージェントオーケストレータは、複数のA2Aエージェントを統合し、LangGraphのcreate_react_agentを使用してインテリジェントなタスク配分と協調実行を行います。

## 主な機能

- **LangGraph統合**: create_react_agentを使用した高度なエージェント制御
- **Gemini 2.5 Flash**: 最新のGoogle Geminiモデルを使用
- **A2Aプロトコル**: 標準化されたエージェント間通信
- **インテリジェントタスク配分**: タスク内容に基づく最適なエージェント選択
- **複雑なワークフロー**: 複数エージェントの協調実行
- **リアルタイム監視**: タスクの状態と進行状況の追跡

## アーキテクチャ

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Orchestrator   │───▶│  A2A Agents     │
│                 │    │  (LangGraph)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Gemini 2.5     │
                       │  Flash LLM      │
                       └─────────────────┘
```

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

### 3. 基本的な使用方法

```python
import asyncio
from host_agent import MultiAgentOrchestrator

async def main():
    # マルチエージェントオーケストレータを作成
    orchestrator = MultiAgentOrchestrator("My Orchestrator")
    
    # エージェントを発見して登録
    agent_urls = [
        "http://localhost:8000",  # 天気エージェント
        "http://localhost:8001",  # 旅行エージェント
        "http://localhost:8002",  # コードエージェント
    ]
    
    for url in agent_urls:
        agent_card = await orchestrator._discover_agent_async(url)
        if agent_card:
            orchestrator.register_agent(agent_card.name, agent_card)
    
    # 複雑なタスクを実行
    result = await orchestrator.orchestrate_complex_task(
        "東京の天気を調べて、良い天気なら旅行の提案をしてください",
        agent_urls
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## クラス構成

### MultiAgentOrchestrator

メインのオーケストレータクラスです。

#### 主要メソッド

- `register_agent(agent_id, agent_card)`: エージェントを登録
- `get_registered_agents()`: 登録されたエージェントの一覧を取得
- `process_user_request(user_message)`: LangGraphワークフローでリクエストを処理
- `orchestrate_complex_task(task_description, agent_urls)`: 複雑なタスクを協調実行

#### 内蔵ツール

1. **discover_agent(agent_url)**: エージェントを発見して登録
2. **list_available_agents()**: 利用可能なエージェントの一覧を取得
3. **send_task_to_agent(agent_name, task_message)**: 指定エージェントにタスクを送信
4. **get_task_status(task_id)**: タスクの状態を取得
5. **select_best_agent_for_task(task_description)**: タスクに最適なエージェントを選択

### A2AClient

A2Aプロトコルに基づくHTTPクライアントです。

#### 主要メソッド

- `get_agent_card()`: エージェントカードを取得
- `send_task(params)`: タスクを送信
- `get_task(task_id)`: タスクの状態を取得

## 使用例

### 1. 基本的なオーケストレーション

```python
async def basic_orchestration():
    orchestrator = MultiAgentOrchestrator()
    
    # エージェントを登録
    agent_card = await orchestrator._discover_agent_async("http://localhost:8000")
    if agent_card:
        orchestrator.register_agent(agent_card.name, agent_card)
    
    # タスクを実行
    result = await orchestrator.process_user_request("天気を教えてください")
    print(result)
```

### 2. 複雑なタスクの協調実行

```python
async def complex_task_execution():
    orchestrator = MultiAgentOrchestrator()
    
    # 複数のエージェントを登録
    agent_urls = [
        "http://localhost:8000",  # 天気エージェント
        "http://localhost:8001",  # 旅行エージェント
    ]
    
    # 複雑なタスクを実行
    task = "東京の天気を調べて、良い天気なら旅行の提案をしてください"
    result = await orchestrator.orchestrate_complex_task(task, agent_urls)
    print(result)
```

### 3. カスタムワークフロー

```python
async def custom_workflow():
    orchestrator = MultiAgentOrchestrator()
    
    # カスタムプロンプトでタスクを実行
    custom_prompt = """
    以下のタスクを実行してください：
    1. 利用可能なエージェントを確認
    2. 最適なエージェントを選択
    3. タスクを送信して結果を取得
    4. 結果をまとめて報告
    
    タスク: 東京の天気と旅行情報を取得
    """
    
    result = await orchestrator.process_user_request(custom_prompt)
    print(result)
```

## LangGraphワークフロー

### 状態管理

```python
class AgentState(TypedDict):
    messages: List[Any]                    # メッセージ履歴
    available_agents: List[Dict[str, Any]] # 利用可能なエージェント
    current_task: Optional[Dict[str, Any]] # 現在のタスク
    task_results: List[Dict[str, Any]]     # タスク結果
```

### ワークフロー制御

```python
def should_continue(state: AgentState) -> str:
    """次のステップを決定する"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # ツール呼び出しがある場合は続行
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "agent"
    
    # それ以外の場合は終了
    return END
```

## エージェント選択ロジック

### キーワードマッチング

```python
keywords = {
    "weather": ["weather", "天気", "気温", "気候"],
    "travel": ["travel", "旅行", "予約", "booking", "hotel"],
    "code": ["code", "programming", "開発", "コード"],
    "analysis": ["analysis", "分析", "データ", "analytics"]
}
```

### スコアリングシステム

- エージェントの説明にキーワードが含まれる: +1点
- タスク内容とエージェントの説明が一致: +2点
- 最高スコアのエージェントが選択される

## 設定オプション

### LLM設定

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Gemini 2.5 Flash
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,               # 創造性の度合い
    max_tokens=2000,               # 最大トークン数
)
```

### タイムアウト設定

```python
class A2AClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.timeout = ClientTimeout(total=timeout)
```

## エラーハンドリング

### 1. イベントループ問題の解決

```python
import nest_asyncio
nest_asyncio.apply()

# 自動エラーハンドリング
try:
    asyncio.run(main())
except RuntimeError as e:
    if "Event loop is closed" in str(e):
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
```

### 2. API キーの確認

```python
if not os.getenv("GOOGLE_API_KEY"):
    print("警告: GOOGLE_API_KEY環境変数が設定されていません。")
    return
```

### 3. エージェント接続エラー

```python
try:
    agent_card = await orchestrator._discover_agent_async(url)
    if agent_card:
        orchestrator.register_agent(agent_card.name, agent_card)
except Exception as e:
    logger.error(f"エージェント接続エラー: {e}")
```

## トラブルシューティング

### 1. 依存関係のインストールエラー

```bash
# 個別にインストール
pip install aiohttp
pip install langchain
pip install langchain-google-genai
pip install google-generativeai
pip install langgraph
pip install langchain-core
pip install pydantic
pip install nest-asyncio
```

### 2. Gemini 2.5 Flashの利用

```python
# モデル名の確認
model="gemini-2.0-flash-exp"  # Gemini 2.5 Flash
```

### 3. A2Aエージェントの設定

```bash
# エージェントサーバーの起動確認
curl http://localhost:8000/.well-known/agent.json
```

## パフォーマンス最適化

### 1. 非同期処理の活用

```python
# 複数エージェントの並行処理
async def discover_multiple_agents(urls):
    tasks = [orchestrator._discover_agent_async(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 2. キャッシュの活用

```python
# エージェントカードのキャッシュ
self.agent_cache = {}

async def get_cached_agent_card(self, url):
    if url not in self.agent_cache:
        self.agent_cache[url] = await self._discover_agent_async(url)
    return self.agent_cache[url]
```

## 注意事項

- この実装はデモンストレーション目的です
- 本番環境では適切なセキュリティ対策を実装してください
- エージェント間の通信は信頼できない入力として扱ってください
- Google API キーは適切に管理してください
- Gemini 2.5 Flashの利用制限を確認してください

## ライセンス

MIT License 
