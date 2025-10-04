import os
import platform
import threading
import time

try:
    import pythoncom
    from win32com.client import Dispatch, GetActiveObject
except ImportError:
    pythoncom = None
    Dispatch = None
    GetActiveObject = None


class _WindowsPowerPointController:
    """Control PowerPoint on Windows via COM."""

    _thread_state = threading.local()
    _RETRYABLE_HRESULTS = {
        0x80010001,  # RPC_E_CALL_REJECTED
        0x8001010A,  # RPC_E_SERVERCALL_RETRYLATER
    }

    def __init__(self):
        if pythoncom is None or Dispatch is None or GetActiveObject is None:
            raise RuntimeError(
                "pywin32 is required to control PowerPoint on Windows."
            )
        self._ensure_thread_com_ready()
        self.app = self._get_application()
        self.presentation = None
        self.show_window = None

    @classmethod
    def _ensure_thread_com_ready(cls):
        if pythoncom is None:
            raise RuntimeError(
                "pywin32 is required to control PowerPoint on Windows."
            )
        if getattr(cls._thread_state, "com_initialized", False):
            return
        try:
            pythoncom.CoInitialize()
        except Exception as exc:
            hresult = getattr(exc, "hresult", None)
            ok_codes = {
                getattr(pythoncom, "S_FALSE", None),
                getattr(pythoncom, "RPC_E_CHANGED_MODE", None),
            }
            if hresult not in ok_codes:
                raise
        cls._thread_state.com_initialized = True

    def _get_application(self):
        try:
            app = GetActiveObject("PowerPoint.Application")
        except Exception:
            app = Dispatch("PowerPoint.Application")
        app.Visible = True
        return app

    def open_file(self, ppt_path: str):
        """Open or reuse a PowerPoint presentation."""
        self._ensure_thread_com_ready()
        abs_path = os.path.abspath(ppt_path)
        presentations = self.app.Presentations

        target = abs_path.lower()
        for pres in presentations:
            try:
                if pres.FullName.lower() == target:
                    self.presentation = pres
                    break
            except Exception:
                continue
        else:
            try:
                self.presentation = self._call_with_retry(
                    presentations.Open,
                    abs_path,
                    False,
                    False,
                    True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to open presentation: {abs_path}"
                ) from exc

        self.show_window = None
        return abs_path

    def start_slideshow(self):
        """Begin the slide show for the active presentation."""
        if self.presentation is None:
            raise RuntimeError("Call open_file before starting a slide show.")

        self._ensure_thread_com_ready()

        existing = self._get_existing_show_window()
        if existing is not None:
            self.show_window = existing
            return

        settings = self.presentation.SlideShowSettings
        self._call_with_retry(settings.Run)

        window = self._wait_for_show_window(timeout=5.0)
        if window is None:
            raise RuntimeError("Failed to start PowerPoint slide show window.")
        self.show_window = window

    def goto_slide(self, index: int):
        """Navigate to the given slide number (1-based)."""
        if index < 1:
            raise ValueError("Slide index must be 1 or greater.")

        self._ensure_thread_com_ready()

        window = self._get_existing_show_window()
        if window is None:
            window = self._wait_for_show_window(timeout=2.0)
        if window is None:
            raise RuntimeError("No active slide show window to control.")
        self.show_window = window

        view = window.View
        self._call_with_retry(view.GotoSlide, index)

    def _wait_for_show_window(self, timeout=5.0, poll_interval=0.2):
        deadline = time.time() + timeout
        while time.time() < deadline:
            window = self._get_existing_show_window()
            if window is not None:
                return window
            time.sleep(poll_interval)
        return None

    def _get_existing_show_window(self):
        candidates = [self.show_window]

        if self.presentation is not None:
            try:
                candidates.append(self.presentation.SlideShowWindow)
            except Exception:
                pass

        for window in candidates:
            if not window:
                continue
            try:
                _ = window.View  # Validate the COM proxy
                return window
            except Exception:
                continue

        try:
            windows = self.app.SlideShowWindows
        except Exception:
            return None

        try:
            return self._call_with_retry(windows.__call__, 1)
        except Exception:
            return None

    def _call_with_retry(self, func, *args, **kwargs):
        last_exc = None
        for _ in range(5):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                hresult = getattr(exc, "hresult", None)
                if hresult in self._RETRYABLE_HRESULTS:
                    time.sleep(0.2)
                    last_exc = exc
                    continue
                raise
        if last_exc is None:
            last_exc = RuntimeError("Unknown failure calling COM method.")
        raise last_exc


class _MacPowerPointController:
    """Control PowerPoint on macOS via AppleScript (osascript)."""

    def __init__(self):
        self.ppt_path = None

    def open_file(self, ppt_path: str):
        """Open a presentation using AppleScript."""
        import subprocess

        self.ppt_path = os.path.abspath(ppt_path)
        script = f'''
        tell application "Microsoft PowerPoint"
            activate
            open POSIX file "{self.ppt_path}"
            delay 0.5
        end tell
        '''
        subprocess.run(["osascript", "-e", script], check=True)
        time.sleep(0.5)

    def start_slideshow(self):
        """Start a slide show using AppleScript."""
        import subprocess

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
        subprocess.run(["osascript", "-e", script], check=True)
        time.sleep(0.5)

    def goto_slide(self, index: int):
        import subprocess
        import textwrap

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
        subprocess.run(["osascript", "-e", script], check=True)
        time.sleep(0.2)


class PowerPointController:
    """
    Factory that returns the appropriate controller for the current platform.

    Methods:
      open_file(path: str)
      start_slideshow()
      goto_slide(index: int)
    """

    def __new__(cls, *args, **kwargs):
        system = platform.system()
        if system == "Windows":
            return _WindowsPowerPointController()
        if system == "Darwin":
            return _MacPowerPointController()
        raise RuntimeError(f"Unsupported OS: {system}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerPoint Controller CLI")
    parser.add_argument("file", help="Path to a PowerPoint file (.pptx)")
    parser.add_argument(
        "-g", "--goto", type=int, default=None, help="Slide number to navigate to"
    )
    args = parser.parse_args()

    controller = PowerPointController()
    controller.open_file(args.file)
    controller.start_slideshow()
    if args.goto is not None:
        controller.goto_slide(args.goto)
    print("PowerPoint automation completed.")
