import subprocess
import sys

n = 30
python_exec = sys.executable

for i in range(n):
    subprocess.run([
    python_exec, "tClassico.py", 
    "1001", "1001", 
    "--min_val", "1", 
    "--max_val", "100", 
    "--seed", "42"
    ])

    subprocess.run([
    python_exec, "tRestrito.py", 
    "1001", "1001", 
    "--min_val", "1", 
    "--max_val", "100", 
    "--seed", "42"
    ])