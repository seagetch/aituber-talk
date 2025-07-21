import platform
import os
import time

# Windows 用 COM 操作用
try:
    import pythoncom
    from win32com.client import Dispatch, GetActiveObject
except ImportError:
    pythoncom = None


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
    macOS 環境で PowerPoint を AppleScript (osascript) で制御する実装
    """
    def __init__(self):
        self.ppt_path = None

    def open_file(self, ppt_path: str):
        """
        指定ファイルをAppleScript経由でPowerPointで開く
        """
        import subprocess
        import time
        self.ppt_path = os.path.abspath(ppt_path)
        script = f'''
        tell application "Microsoft PowerPoint"
            activate
            open POSIX file "{self.ppt_path}"
            delay 0.5
        end tell
        '''
        subprocess.run(["osascript", "-e", script])
        time.sleep(0.5)

    def start_slideshow(self):
        """
        開いたファイルをスライドショーモードで表示（macOS/osascript使用）
        """
        import subprocess
        import time
        script = f'''
        tell application "Microsoft PowerPoint"
            activate
            open POSIX file "{self.ppt_path}"
            delay 0.5
            set thePres to active presentation
            set slideShowSettings to slide show settings of thePres
            run slideShowSettings
            delay 0.5
        end tell
        '''
        subprocess.run(["osascript", "-e", script])
        time.sleep(0.5)

    def goto_slide(self, index: int):
        import subprocess, time, textwrap
        script = textwrap.dedent(f"""
            tell application "Microsoft PowerPoint"
                if (count of slide show windows) is 0 then return
                tell slide show window 1
                    tell its slide show view
                        set curPos to current show position
                        if curPos < {index} then
                            repeat while curPos < {index}
                                go to next slide
                                set curPos to current show position
                            end repeat
                        else if curPos > {index} then
                            repeat while curPos > {index}
                                go to previous slide
                                set curPos to current show position
                            end repeat
                        end if
                    end tell
                end tell
            end tell
        """)
        subprocess.run(["osascript", "-e", script])
        time.sleep(0.2)


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
