import sys
import subprocess
import platform

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    system = platform.system()
    if system == "Windows":
        print("Windows環境: pywin32をインストールします")
        install("pywin32")
    elif system == "Darwin":
        print("macOS環境: pyobjc-framework-Foundation, pyobjc-framework-ScriptingBridgeをインストールします")
        install("pyobjc-framework-Foundation")
        install("pyobjc-framework-ScriptingBridge")
    else:
        print(f"未対応OS: {system}")
        sys.exit(1)
    print("ppt_control.py用の依存パッケージセットアップが完了しました。")

if __name__ == "__main__":
    main()
