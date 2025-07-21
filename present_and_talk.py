#!/usr/bin/env python
import argparse
import re
import time
import requests
from lib.ppt_control import PowerPointController

def parse_script(script_path):
    """script.mdをスライドごとに(タイトル, 本文)リストで返す"""
    slides = []
    with open(script_path, encoding="utf-8") as f:
        content = f.read()
    # "# スライドN"で区切る
    parts = re.split(r'^#\s*スライド\d+.*$', content, flags=re.MULTILINE)
    titles = re.findall(r'^#\s*スライド\d+(.+)$', content, flags=re.MULTILINE)
    # 1ページ目はparts[1]、2ページ目以降はparts[2], ...（parts[0]は空文字列）
    for i, body in enumerate(parts[1:], 1):
        title = titles[i-1].strip() if i-1 < len(titles) else ""
        slides.append((title, body.strip()))
    return slides

def main():
    parser = argparse.ArgumentParser(description="プレゼンを開き、スクリプトに従いページ送り＋読み上げ")
    parser.add_argument("pptx", help="PowerPointファイルのパス")
    parser.add_argument("script", help="スクリプトファイルのパス")
    parser.add_argument("--style_id", type=int, default=None, help="話者のStyleID（省略可）")
    parser.add_argument("--api_url", default="http://127.0.0.1:34512/talk", help="run.pyのAPIエンドポイント")
    parser.add_argument("--wait", type=float, default=1.0, help="ページ送り後の待機秒数")
    args = parser.parse_args()

    # スクリプトをパース
    slides = parse_script(args.script)
    if not slides:
        print("スクリプトが空です")
        return

    # PowerPoint操作
    ctrl = PowerPointController()
    ctrl.open_file(args.pptx)
    ctrl.start_slideshow()
    time.sleep(args.wait)

    # 1ページ目はタイトル無視、本文のみ読み上げ
    first_title, first_body = slides[0]
    # 2ページ目以降
    for idx, (title, body) in enumerate(slides[0:], 1):
        ctrl.goto_slide(idx)
        time.sleep(args.wait)
        if body:
            payload = {"text": body, "sync": True}
            # StyleIDは最初のセリフのみ指定
            requests.post(args.api_url, json=payload)
            print(f"Slide {idx}: 読み上げ送信")
            time.sleep(args.wait)

if __name__ == "__main__":
    main()
