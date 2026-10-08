"""Syntax-check inline HTML scripts with Node when available.

This does not substitute for browser screenshot/layout testing.
"""
from __future__ import annotations
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

def check():
    node=shutil.which("node")
    if not node:
        print("Node unavailable; skipping JS syntax check")
        return
    files=[Path("docs/index.html"),Path("docs/viewer.html")]
    total=0
    with tempfile.TemporaryDirectory() as tmp:
        for path in files:
            text=path.read_text(encoding="utf-8")
            scripts=re.findall(r"<script\b[^>]*>([\s\S]*?)</script\s*>",text,flags=re.I)
            if not scripts:raise ValueError("Expected inline JS in "+str(path))
            for i,source in enumerate(scripts):
                out=Path(tmp)/(path.stem+"_"+str(i)+".js")
                out.write_text(source,encoding="utf-8")
                subprocess.run([node,"--check",str(out)],check=True,capture_output=True,text=True)
                total+=1
    print("JavaScript syntax: VALID across",total,"inline scripts")

if __name__=="__main__":check()
