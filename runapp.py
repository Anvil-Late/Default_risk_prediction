
# -*- coding: utf-8 -*-
import subprocess

import psutil

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

proc = subprocess.Popen(r"streamlit run C:\Users\Antoine\Documents\GitHub\Default_risk_prediction\streamlit_app.py", 
                        shell = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
try:
    outs, errs = proc.communicate(timeout=30)
except subprocess.TimeoutExpired:
    proc.kill()
    outs, errs = proc.communicate()