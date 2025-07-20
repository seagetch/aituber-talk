#!/usr/bin/env python3
"""
app.py

Gradio を使ったシンプルなフロントエンド
 - 話者（style_id）の選択
 - テキスト入力
 - バックエンド /talk へ POST
"""
import gradio as gr
import requests
import os
# AivisSpeech Engine のデフォルト起動ポート: 10101
# AivisSpeechサーバーのURL（環境変数 AIVIS_SPEAKER_URL、未設定時はデフォルト）
SPEAKER_SERVICE_URL = os.getenv("AIVIS_SPEAKER_URL", "http://127.0.0.1:10101")

# バックエンドの URL
BACKEND_URL = "http://127.0.0.1:34512/talk"

# 外部AivisSpeechサーバーから話者一覧を取得
try:
    resp = requests.get(f"{SPEAKER_SERVICE_URL}/speakers")
    resp.raise_for_status()
    # VOICEVOX互換の /speakers は speaker.name と styles[] 配列を返す仕様のためフラット化
    raw_speakers = resp.json()
except Exception as e:
    print(f"Failed to fetch speakers from {SPEAKER_SERVICE_URL}/speakers: {e}")
    # フォールバックのデフォルト話者
    raw_speakers = [{"name": "デフォルト", "styles": [{"id": 888753760, "name": "デフォルト"}]}]

def update_styles(speaker_name):
    for sp in raw_speakers:
        if sp.get("name") == speaker_name:
            choices = [f"{style.get('name')} ({style.get('id')})" for style in sp.get("styles", [])]
            default = choices[0] if choices else None
            return gr.update(choices=choices, value=default)
    return gr.update(choices=[], value=None)

def send_text(text: str, style_choice: str) -> str:
    # style_choice は "名前 (ID)" 形式
    try:
        style_id = int(style_choice.split("(")[-1].strip(")"))
    except Exception:
        style_id = raw_speakers[0]["styles"][0]["id"]
    payload = {"text": text, "style_id": style_id}
    try:
        resp = requests.post(BACKEND_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return f"ステータス: {data.get('status')}"
    except Exception as e:
        return f"エラー: {e}"

# Gradio UI 定義
with gr.Blocks(title="テキスト→音声フロントエンド") as demo:
    gr.Markdown("## テキスト音声化デモ")
    with gr.Row():
        speaker_dropdown = gr.Dropdown(
            choices=[sp.get("name") for sp in raw_speakers],
            label="スピーカーを選択",
            value=raw_speakers[0].get("name")
        )
        style_dropdown = gr.Dropdown(
            choices=[f"{style.get('name')} ({style.get('id')})" for style in raw_speakers[0].get("styles", [])],
            label="スタイルを選択",
            value=f"{raw_speakers[0]['styles'][0].get('name')} ({raw_speakers[0]['styles'][0].get('id')})"
        )
        speaker_dropdown.change(
            update_styles,
            inputs=[speaker_dropdown],
            outputs=[style_dropdown]
        )
        text_input = gr.Textbox(
            label="テキスト入力",
            placeholder="ここにテキストを入力",
            lines=4
        )
    send_btn = gr.Button("送信")
    status_output = gr.Textbox(label="レスポンスステータス", interactive=False)

    send_btn.click(
        send_text,
        inputs=[text_input, style_dropdown],
        outputs=status_output
    )

if __name__ == "__main__":
    # Gradio サーバー起動
    demo.launch(server_name="0.0.0.0", server_port=8000)
