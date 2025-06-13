import sys
import os
import subprocess
import shutil
import torch

TOOLS_DIR = os.environ.get("YOLO_TOOLS_DIR", "/workspace/tools")
os.makedirs(TOOLS_DIR, exist_ok=True)

def run_cmd(cmd, cwd=None):
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def ensure_yolov5():
    path = os.path.join(TOOLS_DIR, "yolov5")
    if not os.path.exists(path):
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5", path])
        run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=path)
    return path

def ensure_yolov7():
    path = os.path.join(TOOLS_DIR, "yolov7")
    if not os.path.exists(path):
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/WongKinYiu/yolov7", path])
        run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=path)
    return path

def ensure_yolov8():
    path = os.path.join(TOOLS_DIR, "ultralytics")
    if not os.path.exists(path):
        run_cmd([sys.executable, "-m", "pip", "install", "ultralytics"])
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: convert.py <model.pt> <version: v5|v6|v7|v8>")
        sys.exit(1)

    model_path = sys.argv[1]
    version = sys.argv[2].lower()

    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        sys.exit(1)

    if version == "v5":
        yolov5_path = ensure_yolov5()
        run_cmd([
            sys.executable, "export.py",
            "--weights", model_path,
            "--img", "640", "--batch", "1",
            "--include", "onnx"
        ], cwd=yolov5_path)

    elif version == "v7":
        yolov7_path = ensure_yolov7()
        run_cmd([
            sys.executable, "export.py",
            "--weights", model_path,
            "--img", "640", "--batch", "1",
            "--dynamic", "--simplify"
        ], cwd=yolov7_path)

    elif version == "v8":
        ensure_yolov8()
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format='onnx', imgsz=[640, 640], simplify=True, dynamic=True)

    else:
        print("Unsupported version. Use v5, v7, or v8.")
        sys.exit(1)

if __name__ == "__main__":
    main()
