import os
import shutil
import subprocess

REQUIRED_DIRS = [
    "src/config",
    "src/llm_providers",
    "src/rag",
    "src/evaluation",
    "src/data_processing",
    "src/ui",
    "src/security",
    "src/utils",
    "config",
    "data",
    "docs",
    "tests",
    "scripts",
]

def create_dirs():
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)

def copy_env():
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        shutil.copy(".env.example", ".env")
        print(".env file created from .env.example")
    else:
        print(".env already exists or .env.example missing.")

def install_deps():
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    if os.path.exists("requirements-dev.txt"):
        subprocess.run(["pip", "install", "-r", "requirements-dev.txt"])

def main():
    create_dirs()
    copy_env()
    print("Setup complete. Please activate your virtual environment if not already done.")

if __name__ == "__main__":
    main() 