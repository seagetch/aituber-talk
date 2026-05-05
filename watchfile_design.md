# Codex Watch Talk Mode Design

## 目的

`aituber-talk` に、Codex のローカルセッション JSONL を監視し、Codex の更新内容をキャラクターのセリフへ変換して `TalkEngine` で発話させる新しい Mode Plugin を追加する。

この Mode は Codex 内部ファイルを読み取り専用で扱う。`~/.codex/sessions`、`~/.codex/session_index.jsonl`、`~/.codex/sqlite` などへは書き込まない。

## 確認済みの前提

この環境では以下が確認できる。

- session index:
  - `~/.codex/session_index.jsonl`
  - 各行に `id`, `thread_name`, `updated_at` がある。
- rollout JSONL:
  - `~/.codex/sessions/**/*.jsonl`
  - ファイル名は `rollout-*.jsonl`
  - `session_meta.payload.id` が session id
  - `session_meta.payload.cwd` が作業ディレクトリ
- 監視対象 event:
  - `event_msg.user_message`
  - `event_msg.agent_message`
  - `response_item.message`
  - 必要に応じて `response_item.function_call`, `response_item.function_call_output`

session 一覧 UI では、`session_index.jsonl` の `thread_name` をタイトルに使い、rollout JSONL の `session_meta.payload.cwd` をプロジェクト表示に使う。

## 新 Mode

Mode 名は `codex_watch` とする。

役割は以下。

1. Codex session 一覧を UI に提供する。
2. UI で選択された session の `rollout-*.jsonl` を監視する。
3. JSONL 追記分から表示対象のメッセージ更新だけを抽出する。
4. OpenAI API で、抽出メッセージをキャラクターのセリフへ変換する。
5. 生成したセリフを `TalkEngine.submit_text()` に渡す。

Mode Plugin は TTS、SadTalker、OSC へ直接触らない。発話とモデル転送は既存の `TalkEngine` に集約する。

## UI 仕様

Codex Watch タブを追加する。

入力:

- Codex session dropdown
  - 表示例: `確認 app.pyで画像付きプレゼン | aituber-talk | 2026-05-05 09:19`
  - value は rollout JSONL path か session id
- session details
  - session id
  - thread name
  - cwd
  - updated_at
  - rollout path
- Character profile textarea
  - 性格
  - 口調
  - 一人称
  - ユーザーへの呼び方
  - 禁止事項
- Style id
  - `TalkEngine.submit_text(style_id=...)` へ渡す。
- OpenAI model
  - default: `gpt-5.4-nano`
  - UI で変更可能にする。
- Watch event types
  - 初期値: assistant-visible messages only
  - `event_msg.agent_message`
  - `response_item.message role=assistant`
- Start Watching
- Stop Watching

表示:

- Watch status
  - `running`, `stopped`, `error`
- Last Codex message
- Last generated utterance
- Recent utterance history
- Error log

## メッセージパッシング

全体フロー:

```text
UI
  -> POST /sessions mode=codex_watch
  -> controller.api.create_session()
  -> SessionManager.create()
  -> CodexWatchMode.start()
  -> CodexJsonlWatcher thread
  -> Codex event extraction
  -> CodexWatchMode speech queue
  -> OpenAI rewrite worker
  -> TalkEngine.submit_text()
  -> TalkEngine._text_queue
  -> TTS / SadTalker
  -> TalkEngine._motion_queue
  -> TalkEngine._consumer_loop
  -> OSC/VMC
```

`EventBus` は UI への状態通知に使う。

発行イベント:

```text
codex_watch.started
codex_watch.session_selected
codex_watch.codex_message
codex_watch.rewrite_started
codex_watch.rewrite_completed
codex_watch.speech_queued
codex_watch.error
codex_watch.stopped
```

## モジュール構成

### `core/codex_sessions.py`

Codex session の一覧化だけを担当する。

責務:

- `~/.codex/session_index.jsonl` を読む。
- `~/.codex/sessions/**/rollout-*.jsonl` を探す。
- rollout JSONL の先頭付近から `session_meta` を読む。
- `id`, `thread_name`, `updated_at`, `cwd`, `rollout_path`, `mtime` を統合する。

返却型案:

```python
@dataclass
class CodexSessionInfo:
    id: str
    thread_name: str
    updated_at: str
    cwd: str | None
    rollout_path: Path
    mtime: float
```

### `core/codex_watcher.py`

JSONL 監視と増分 parse だけを担当する。Controller、UI、OpenAI、TalkEngine に依存しない。

責務:

- `watchfiles.watch()` で `rollout-*.jsonl` を監視する。
- ファイルごとに `offset` と `partial_buffer` を持つ。
- 変更通知は wakeup signal として扱う。
- offset 以降を読み、改行で閉じた JSONL 行だけ parse する。
- 不完全行は `partial_buffer` に残す。
- truncate/rotate された場合は offset を 0 に戻す。

抽出対象は正規化して返す。

```python
@dataclass
class CodexMessageEvent:
    session_id: str | None
    session_file: Path
    timestamp: str | None
    source_type: str
    role: str | None
    text: str
    cwd: str | None
    raw: dict[str, Any]
```

初期実装では以下だけを発話候補にする。

- `event_msg.agent_message`
- `response_item.message` where `role == "assistant"`

`event_msg.user_message` は履歴には入れるが、初期実装では発話トリガーにしない。

### `plugins/codex_watch.py`

Mode Plugin 本体。

責務:

- UI payload を受け取り、対象 session/rollout path を決定する。
- watcher thread を開始/停止する。
- Codex message history を保持する。
- generated utterance history を保持する。
- OpenAI rewrite worker に投入する。
- 生成セリフを `TalkEngine.submit_text()` に渡す。

内部 queue を分ける。

```text
CodexJsonlWatcher
  -> message_queue
  -> rewrite_worker
  -> TalkEngine.submit_text()
```

`watchfiles` callback や JSONL parse loop 内で OpenAI API や `wait_for()` を呼ばない。

## OpenAI 変換仕様

環境変数:

```text
OPENAI_API_KEY
```

`.env` は `python-dotenv` で読み込む。ただし `.env` は git 管理しない。

モデル:

```text
default: gpt-5.4-nano
```

UI から model name を変更可能にする。

OpenAI 入力:

```text
system:
  あなたはVTuberキャラクターのセリフ変換器。
  入力されたCodex更新を、短い自然な日本語セリフに変換する。
  説明ではなく、キャラクターが今喋る一言だけを出す。

character_profile:
  UIで入力された性格設定

recent_codex_messages:
  直近N件のCodex表示メッセージ

recent_character_utterances:
  直近N件の生成済みセリフ

new_codex_update:
  今回抽出したCodex message
```

OpenAI 出力は JSON に固定する。

```json
{
  "should_speak": true,
  "utterance": "短いセリフ"
}
```

`should_speak=false` の条件:

- 空メッセージ
- 重複メッセージ
- tool output だけ
- 内部ログ
- 長すぎてセリフ化に向かない更新

セリフ長は初期実装では 120 文字程度を上限目安にする。

## 履歴管理

`CodexWatchMode` は session ごとに以下を保持する。

- `recent_codex_messages`: 最大 20 件
- `recent_character_utterances`: 最大 20 件
- `last_offsets`: watcher が持つ offset
- `last_seen_message_hashes`: 重複発話抑制用

再起動後の offset 永続化は第 2 段階でよい。

初期実装では、watch 開始時に対象 rollout の末尾から監視を始める。既存ログを大量に読んで発話しない。

## Controller API

既存 `/sessions` に mode として載せる。

開始 payload:

```json
{
  "mode": "codex_watch",
  "payload": {
    "session_id": "019df76e-ce52-7760-8c2e-d417b5bcdf11",
    "rollout_path": "/Users/seagetch/.codex/sessions/2026/05/05/rollout-....jsonl",
    "character_profile": "...",
    "style_id": 888753760,
    "model": "gpt-5.4-nano",
    "event_types": ["assistant_message"]
  }
}
```

`status()` 返却:

```json
{
  "session": "...",
  "watch_status": "running",
  "codex_session_id": "...",
  "thread_name": "...",
  "cwd": "...",
  "rollout_path": "...",
  "last_codex_message": "...",
  "last_utterance": "...",
  "messages_seen": 12,
  "utterances_generated": 4,
  "last_error": null
}
```

追加 API は第 1 段階では作らない。Codex session 一覧は mode metadata では表現しにくいため、必要なら第 1 段階で以下を追加する。

```text
GET /codex/sessions
```

返却:

```json
{
  "sessions": [
    {
      "id": "...",
      "thread_name": "...",
      "updated_at": "...",
      "cwd": "...",
      "rollout_path": "..."
    }
  ]
}
```

## uv による環境管理

今後は依存追加と実行を `uv` で行う。

まず `pyproject.toml` を追加し、既存 `requirements.txt` から移行する。

初期追加依存:

```toml
[project]
dependencies = [
  "fastapi",
  "uvicorn",
  "gradio",
  "requests",
  "openai",
  "python-dotenv",
  "watchfiles",
]
```

実際には既存 `requirements.txt` の全依存を `pyproject.toml` に反映してから移行する。SadTalker、MediaPipe、PyQt、pygame など既存機能に必要な依存を落とさない。

コマンド:

```bash
uv init --bare
uv add openai python-dotenv watchfiles
uv sync
uv run python app.py --ui pyqt
uv run python -m unittest
```

既存 `.venv` を使っている場合も、今後の追加作業では `uv add` と `uv run` を標準にする。

`.env` は `.gitignore` に追加する。

```text
.env
```

## テスト計画

Unit tests:

- `core/codex_sessions.py`
  - `session_index.jsonl` を読める。
  - rollout JSONL から `session_meta` を読める。
  - thread name と cwd を統合できる。
- `core/codex_watcher.py`
  - 完全行を parse できる。
  - 分割書き込みされた行を buffer できる。
  - truncate 時に offset を 0 に戻せる。
  - 不正 JSON 行で watcher が停止しない。
- `plugins/codex_watch.py`
  - Codex message を受けて OpenAI client mock を呼ぶ。
  - `should_speak=true` なら `TalkEngine.submit_text()` が呼ばれる。
  - `should_speak=false` なら発話しない。
  - stop で watcher thread と worker thread が止まる。

Integration tests:

- 一時ディレクトリに `rollout-*.jsonl` を作り、追記を検出する。
- controller の `/codex/sessions` が session 一覧を返す。
- mode start 後に JSONL 追記で `codex_watch.codex_message` が EventBus に流れる。

Manual verification:

- Codex 実セッションを選択できる。
- assistant message 追記後、数秒以内に UI に表示される。
- OpenAI 変換後、`TalkEngine` から発話される。
- Stop 後に追記しても発話しない。

## 実装順

1. `pyproject.toml` 追加、`uv` 管理開始。
2. `.gitignore` に `.env` を追加。
3. `core/codex_sessions.py` 実装。
4. `core/codex_watcher.py` 実装。
5. `plugins/codex_watch.py` 実装。
6. `controller/app.py` に mode 登録。
7. 必要なら `controller/api.py` に `GET /codex/sessions` 追加。
8. Web UI に Codex Watch タブ追加。
9. PyQt UI に Codex Watch タブ追加。
10. OpenAI client mock 付き unit tests。
11. 実 OpenAI API で manual verification。

## 非目標

初期実装では以下をやらない。

- Codex 内部 DB や session JSONL への書き込み
- workspace diff の読み上げ
- tool output の全文読み上げ
- 複数 Codex session の同時監視
- 発話中断/割り込みの高度な優先度制御
- OpenAI 以外のLLM provider対応
