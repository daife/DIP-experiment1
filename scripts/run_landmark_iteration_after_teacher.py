"""Wait for the specific live Windows teacher process, then run iterations."""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pid = int((ROOT/'tmp/landmark_iteration/teacher.pid').read_text().strip())
kernel = ctypes.windll.kernel32
kernel.OpenProcess.restype = ctypes.c_void_p
kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
kernel.CloseHandle.argtypes = [ctypes.c_void_p]
handle = kernel.OpenProcess(0x1000, False, pid)
if not handle:
    raise RuntimeError(f'Teacher handle {pid} is missing; inspect logs before retrying')
try:
    while True:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise RuntimeError('Cannot query teacher process')
        if code.value != 259:
            if code.value != 0:
                raise RuntimeError(f'Teacher failed: exit {code.value}')
            break
        time.sleep(10)
finally:
    kernel.CloseHandle(handle)
subprocess.run([sys.executable, str(ROOT/'scripts/iterate_landmark_teacher.py'),
                '--run', str(ROOT/'datasets/annotations/auto/hrnetv2_full_20260927')],
               cwd=ROOT, check=True)
