# aituber-talk

aituber-talkは、**テキストを入力するとAIアバターがその内容を発話し、顔の動きやリップシンクを伴うモーションをVMCプロトコルで送信する**AIアバター生成パイプラインです。

本プロジェクトは
- [aivisspeech](https://github.com/seagetch/aivisspeech)（テキスト→音声合成/TTS）
- [SadTalker](https://github.com/OpenTalker/SadTalker)（音声＋画像→顔アニメーション生成）
を統合し、テキスト入力から音声合成、リップシンク、表情生成までを一貫して自動化します。

生成されたモーションはVMCプロトコルで外部のVRMアバターなどにリアルタイムで送信できます。

**用途例:**
- テキスト原稿から自動でAIアバターによるプレゼン動画を作成
- デスクトップアプリやWeb UIでテキストを入力し、即座にVRMアバターを動かす
- API経由で自動動画生成パイプラインを構築

---

## 主な特徴

- **統合アプリケーション:** `app.py`を実行するだけで、バックエンドとUIが一体となって起動します。
- **選べるUI:** 用途に応じて2種類のUIを選択できます。
  - **PyQtデスクトップアプリ (デフォルト):** ローカルでの利用に最適化された、応答性の高いUI。
  - **Gradio Web UI:** ブラウザ経由で手軽にアクセスできるUI。
- **リアルタイムモーション送信:** 生成された顔のモーションをVMCプロトコルで送信し、VRMアバターなどをリアルタイムに動かせます。
- **高品質なAIモデル:** aivisspeechによる高品質な音声合成と、SadTalkerによる高精度な顔アニメーション生成を統合。
- **プレゼンテーションモード:** PowerPointと連携し、スライドのノートを自動で読み上げてプレゼンテーションを自動進行させる機能を搭載。

---

## システムアーキテクチャ

`app.py`が単一のエントリポイントとなり、バックエンドとなる「コントローラーサービス」と、操作用の「UI」を起動します。

```mermaid
flowchart TD
    subgraph User Interface
        direction LR
        user_desktop["User (Desktop App)"]
        user_web["User (Web Browser)"]
    end

    subgraph AITuber Talk Application
        app_py["app.py --ui [pyqt|web]"]
        controller["Controller Service"]
        pyqt_ui["PyQt UI"]
        web_ui["Web UI (Gradio)"]
    end

    subgraph External Services
        aivisspeech["aivisspeech (TTS)"]
        vmc_receiver["Avatar App (VMC Receiver)"]
    end

    user_desktop --> pyqt_ui
    user_web --> web_ui

    app_py --> controller
    app_py --> pyqt_ui
    app_py --> web_ui

    pyqt_ui -->|HTTP API| controller
    web_ui -->|HTTP API| controller

    controller -->|Generates Motion| SadTalker
    controller -->|HTTP API| aivisspeech
    controller -->|VMC Protocol| vmc_receiver
```

---

## インストール

環境構築用のスクリプトが用意されています。お使いの環境に合わせて実行してください。

### 1. Python環境の構築

プロジェクトに必要なライブラリを含むPython仮想環境を構築します。

**Windows (PowerShell):**
```powershell
.\scripts\setup_env.ps1
```

**Linux/macOS (Bash):**
```bash
bash ./scripts/setup_env.sh
```

スクリプトが完了すると、プロジェクトルートに`.venv`という名前の仮想環境が作成されます。

### 2. SadTalkerモデルのダウンロード

顔アニメーション生成に必要な学習済みモデルをダウンロードします。

**Windows (PowerShell):**
```powershell
.\scripts\setup_sadtalker.ps1
```

**Linux/macOS (Bash):**
```bash
bash ./scripts/setup_sadtalker.sh
```

### 3. (任意) aivisspeechのセットアップ

高品質な日本語音声合成を利用するには、別途[aivisspeech](https://github.com/seagetch/aivisspeech)のセットアップが必要です。手順に従ってDockerコンテナを起動してください。


---

## 使い方

`app.py`を実行することで、バックエンドのコントローラーサービスとUIが同時に起動します。

### PyQtデスクトップUI (デフォルト)

コマンドプロンプトやターミナルで以下を実行します。

```bash
python app.py
```

応答性の高いデスクトップアプリケーションが起動します。

### Gradio Web UI

Webブラウザで利用したい場合は、`--ui web`オプションを付けて実行します。

```bash
python app.py --ui web
```

起動後、ターミナルに表示されるURL (`http://127.0.0.1:8000`など) にブラウザでアクセスしてください。

### 主なコマンドライン引数

- `--ui [pyqt|web]`: 起動するUIを選択します (デフォルト: `pyqt`)。
- `--device [auto|cpu|cuda]`: AIモデルを実行するデバイスを選択します (デフォルト: `auto`)。
- `--list-devices`: 利用可能なデバイスの一覧を表示して終了します。
- `--open-browser`: Web UI利用時に、起動と同時にブラウザを自動で開きます。

---

## ディレクトリ構成

- `app.py`: **統合アプリケーションランチャー (メインの実行ファイル)**
- `requirements.txt`: 依存パッケージリスト
- `ui/`: UI関連のコード
  - `pyqt/app.py`: PyQtデスクトップアプリケーションの実装
  - `web/app.py`: Gradio Web UIの実装
- `controller/`: バックエンドのコントローラーサービス関連のコード
- `core/`: `TalkEngine`など、中核となるAI処理パイプライン
- `SadTalker/`: SadTalker本体および関連ファイル
- `scripts/`: セットアップ用のスクリプト

## ライセンス

本プロジェクトはSadTalkerのライセンス（[SadTalker/LICENSE](SadTalker/LICENSE)）に準拠します。詳細は各サブディレクトリのLICENSEファイルを参照してください。