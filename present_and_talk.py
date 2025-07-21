#!/usr/bin/env python
import argparse
import re
import time
import requests
from lib.ppt_control import PowerPointController
 
from io import BytesIO

import pygame
from io import BytesIO
from PIL import Image
import threading
import sys

# Use a Chrome-like User-Agent for requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}

def parse_script(script_path):
    """script.mdをスライドごとに(タイトル, 本文)リストで返す"""
    slides = []
    with open(script_path, encoding="utf-8") as f:
        lines = f.readlines()

    current_title = ""
    current_body_lines = []
    for line in lines:
        m = re.match(r'^(#+)\s*(.*)', line)
        if m:
            # 新しいセクション開始
            if current_title or current_body_lines:
                slides.append((current_title, ''.join(current_body_lines).strip()))
            current_title = m.group(2).strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)
    # 最後のセクションを追加
    if current_title or current_body_lines:
        slides.append((current_title, ''.join(current_body_lines).strip()))

    return slides


class ImageViewer:
    """Simple resizable image viewer using pygame."""
    def __init__(self, title="Image Viewer"):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.original = None

    def show_image(self, pil_img):
        """Display a PIL.Image, resizing it to fit while keeping aspect ratio."""
        mode = pil_img.mode
        size = pil_img.size
        data = pil_img.tobytes()
        self.original = pygame.image.fromstring(data, size, mode)
        self._update_display()

    def _update_display(self):
        # Handle events and redraw
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
        if self.original:
            w, h = self.screen.get_size()
            sw, sh = self.original.get_size()
            ratio = min(w/sw, h/sh)
            new_size = (int(sw*ratio), int(sh*ratio))
            img = pygame.transform.smoothscale(self.original, new_size)
            self.screen.fill((0, 0, 0))
            rect = img.get_rect(center=(w//2, h//2))
            self.screen.blit(img, rect)
            pygame.display.flip()

def main():
    parser = argparse.ArgumentParser(description="プレゼンを開き、スクリプトに従いページ送り＋読み上げ")
    parser.add_argument("script", help="スクリプトファイルのパス")
    parser.add_argument("--ppt", dest="pptx", default=None, help="PowerPointファイルのパス（省略可）")
    parser.add_argument("--style_id", type=int, default=None, help="話者のStyleID（省略可）")
    parser.add_argument("--api_url", default="http://127.0.0.1:34512/talk", help="run.pyのAPIエンドポイント")
    parser.add_argument("--wait", type=float, default=1.0, help="ページ送り後の待機秒数")
    parser.add_argument("--img", action="store_true", help="画像ビューアを起動してMarkdown内の画像を表示")
    args = parser.parse_args()

    # スクリプトをパース
    slides = parse_script(args.script)
    if not slides:
        print("スクリプトが空です")
        return

    # PowerPoint操作（pptxが指定された場合のみ）
    ctrl = None
    if args.pptx:
        ctrl = PowerPointController()
        ctrl.open_file(args.pptx)
        ctrl.start_slideshow()
        time.sleep(args.wait)

    viewer = ImageViewer() if args.img else None

    # スライド処理関数（ワーカースレッドで実行）
    def slide_runner():
        for idx, (title, body) in enumerate(slides, 1):
            if ctrl:
                ctrl.goto_slide(idx)
            if viewer:
                # Markdown 画像タグ ![alt](url) を探す
                m_img = re.search(r'!\[.*?\]\((.*?)\)', body)
                if m_img:
                    img_src = m_img.group(1)
                    try:
                        if re.match(r'^https?://', img_src):
                            resp = requests.get(img_src, headers=HEADERS)
                            resp.raise_for_status()
                            img_bytes = resp.content
                        else:
                            with open(img_src, "rb") as f:
                                img_bytes = f.read()
                        from PIL import Image
                        pil_img = Image.open(BytesIO(img_bytes))
                        pil_img.load()
                        viewer.original = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, pil_img.mode)
                    except Exception as e:
                        print(f"画像処理失敗 ({img_src}): {e}")
            # 画像・リンクタグ除去して読み上げ
            clean_body = re.sub(r'!?\[.*?\]\(.*?\)', '', body).strip()
            if clean_body:
                print(f"Slide {idx}: 読み上げ送信")
                requests.post(args.api_url, json={"text": clean_body, "sync": True})
                print(f"Slide {idx}: 読み上げ完了")
            time.sleep(args.wait)
    # ワーカースレッド起動
    if viewer:
        runner = threading.Thread(target=slide_runner, daemon=True)
        runner.start()
        # メインスレッドでUIイベントを処理
        while runner.is_alive():
            viewer._update_display()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            time.sleep(0.01)
        pygame.quit()
    else:
        slide_runner()

if __name__ == "__main__":
    main()
