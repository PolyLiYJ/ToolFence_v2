"""Run the clean evaluation locally, forwarding only Qwen requests over SSH."""
import argparse
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import getproxies


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file",required=True,type=Path)
    parser.add_argument("--agentdojo-src",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--grant-scope",choices=["all","tool","source","stage","budget"],default="all")
    parser.add_argument("--compiler-model",choices=["gemini","qwen"],default="gemini")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    key=args.key_file.read_text().strip()
    if not re.fullmatch(r"AIza[\w-]{35}",key):
        raise SystemExit("Expected one Google API key in the local key file.")
    proxies=getproxies()
    proxy=proxies.get("https") or proxies.get("http")
    if not proxy:
        raise SystemExit("No local HTTPS proxy configured.")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    tunnel=subprocess.Popen(["ssh","-N","-o","BatchMode=yes","-o","ExitOnForwardFailure=yes",
        "-o","ConnectTimeout=10","-L",f"127.0.0.1:{port}:127.0.0.1:8001","yjli@10.21.4.20"])
    try:
        for _ in range(80):
            if tunnel.poll() is not None:
                raise SystemExit("SSH forward failed")
            try:
                with socket.create_connection(("127.0.0.1",port),timeout=.25): break
            except OSError: time.sleep(.25)
        else: raise SystemExit("SSH forward timed out")
        env={**os.environ,"GEMINI_API_KEY":key,"HTTPS_PROXY":proxy,"https_proxy":proxy,
            "NO_PROXY":"127.0.0.1,localhost","no_proxy":"127.0.0.1,localhost",
            "TOOLFENCE_QWEN_URL":f"http://127.0.0.1:{port}/v1",
            "PYTHONPATH":os.pathsep.join(map(str,[root/"src",args.agentdojo_src.resolve()]))}
        cmd=[sys.executable,str(root/"experiments/grant_scope_clean.py"),"--output",str(args.output.resolve()),
             "--grant-scope",args.grant_scope,"--compiler-model",args.compiler_model]
        if args.pilot: cmd.append("--pilot")
        raise SystemExit(subprocess.call(cmd,env=env))
    finally:
        tunnel.terminate()
        tunnel.wait()

if __name__=="__main__":
    main()
