import sys
import os

print("Hello from check_python.py!")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")