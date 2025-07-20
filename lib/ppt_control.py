import platform
import os
import time

# Windows 用 COM 操作用
try:
    import pythoncom
    from win32com.client import Dispatch, GetActiveObject
except ImportError:
    pythoncom = None

# macOS 用 ScriptingBridge 用
try:
    from Foundation import NSURL
    from ScriptingBridge import SBApplication
except ImportError:
    SBApplication = None


class _WindowsPowerPointController:
    """
    Windows 環境で PowerPoint を COM 経由で制御する実装
    """
    def __init__(self):
        pythoncom.CoInitialize()
        try:
            # 既存プロセスにアタッチ
            self.app = GetActiveObject("PowerPoint.Application")
        except Exception:
            # なければ新規起動
            self.app = Dispatch("PowerPoint.Application")
        self.app.Visible = True

    def open_file(self, ppt_path: str):
        """
        指定ファイルを既存または新規 PowerPoint プロセスで開く
        """
        abs_path = os.path.abspath(ppt_path)
        # WithWindow=True でウィンドウ有効
        self.presentation = self.app.Presentations.Open(abs_path, WithWindow=True)
        time.sleep(0.5)

    def start_slideshow(self):
        """
        開いたファイルをスライドショーモードで表示
        """
        settings = self.presentation.SlideShowSettings
        settings.Run()
        time.sleep(0.5)
        # SlideShowWindow を取得
        self.show_window = self.app.SlideShowWindows(1)

    def goto_slide(self, index: int):
        """
        スライドショーモード中に任意のスライド番号へジャンプ
        index: 1 始まりのスライド番号
        """
        self.show_window.View.GotoSlide(index)


class _MacPowerPointController:
    """
    macOS 環境で PowerPoint を ScriptingBridge 経由で制御する実装
    """
    BUNDLE_ID = "com.microsoft.Powerpoint"

    def __init__(self):
        # 既存プロセスへアタッチ
        self.ppt = SBApplication.applicationWithBundleIdentifier_(self.BUNDLE_ID)

    def open_file(self, ppt_path: str):
        """
        指定ファイルを既存 PowerPoint プロセスで開く
        """
        url = NSURL.fileURLWithPath_(ppt_path)
        self.presentation = self.ppt.open_(url)
        time.sleep(0.5)

    def start_slideshow(self):
        """
        開いたファイルをスライドショーモードで表示
        """
        # AppleScript 辞書に合わせたメソッド
        self.ppt.startSlideShow_(self.presentation)
        time.sleep(0.5)
        # slideShowView を保持
        self.show_view = self.presentation.slideShowView()

    def goto_slide(self, index: int):
        """
        スライドショーモード中に任意のスライド番号へジャンプ
        index: 1 始まりのスライド番号
        """
        self.show_view.goToSlideIndex_(index)


class PowerPointController:
    """
    Windows / macOS を判定して適切なコントローラを返すファクトリクラス

    メソッド:
      open_file(path: str)
      start_slideshow()
      goto_slide(index: int)
    """
    def __new__(cls, *args, **kwargs):
        system = platform.system()
        if system == "Windows":
            if pythoncom is None:
                raise RuntimeError("pywin32 がインストールされていません。Windows での操作には pywin32 が必要です。")
            return _WindowsPowerPointController()
        elif system == "Darwin":
            if SBApplication is None:
                raise RuntimeError("pyobjc がインストールされていません。macOS での操作には pyobjc が必要です。")
            return _MacPowerPointController()
        else:
            raise RuntimeError(f"Unsupported OS: {system}")


if __name__ == "__main__":
    # テスト実行例
    import argparse

    parser = argparse.ArgumentParser(description="PowerPoint Controller テスト")
    parser.add_argument('file', help='PowerPoint ファイルのパス (.pptx)')
    parser.add_argument('-g', '--goto', type=int, default=None, help='ジャンプするスライド番号')
    args = parser.parse_args()

    ctrl = PowerPointController()
    ctrl.open_file(args.file)
    ctrl.start_slideshow()
    if args.goto is not None:
        ctrl.goto_slide(args.goto)
    print("操作が完了しました。")