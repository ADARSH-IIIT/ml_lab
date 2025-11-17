import importlib
import subprocess
import sys
import shutil

def check_module(name):
    try:
        module = importlib.import_module(name)
        print(f"✔ {name} is installed (version: {module.__version__ if hasattr(module, '__version__') else 'OK'})")
    except ImportError:
        print(f"✖ {name} is NOT installed")

def check_conda():
    print("\n=== Checking Conda ===")
    conda = shutil.which("conda")
    if conda:
        print(f"✔ Conda found at: {conda}")
        try:
            version = subprocess.check_output(["conda", "--version"], stderr=subprocess.STDOUT).decode()
            print("✔ " + version.strip())
        except Exception as e:
            print("✖ Conda exists but failed to get version:", e)
    else:
        print("✖ Conda NOT found in PATH")

def check_gpu():
    print("\n=== Checking GPU Support ===")
    try:
        import torch
        print(f"✔ PyTorch version: {torch.__version__}")
        print("✔ CUDA available in PyTorch:" , torch.cuda.is_available())
    except ImportError:
        print("✖ PyTorch NOT installed")

    try:
        import tensorflow as tf
        print(f"✔ TensorFlow version: {tf.__version__}")
        print("✔ GPU devices:", tf.config.list_physical_devices('GPU'))
    except ImportError:
        print("✖ TensorFlow NOT installed")

print("=== Checking Python Environment ===")
print(f"Python version: {sys.version}\n")

print("=== Checking Common ML Libraries ===")
modules = [
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "scipy",
    "jupyter"
]

for m in modules:
    check_module(m)

check_conda()
check_gpu()

print("\nAll checks complete!") 



