"""
不打开任何 App，亲眼看看 App 和 MCP 服务器之间到底在说什么
==============================================================

Claude Desktop 连接本地 MCP 服务器的方式是：把服务器当成子进程启动，
然后通过 stdin / stdout 一行一行地收发 JSON（JSON-RPC 2.0 格式）。

这个脚本就扮演"App"的角色，做完全一样的事情：
  1. 启动两个 bridge_server.py 进程：一个代表 Claude，一个代表 ChatGPT
  2. 和它们握手（initialize），列出工具（tools/list），调用工具（tools/call）
  3. 让 Claude 写一封信，ChatGPT 收信、回信，Claude 再收信

运行：python wire_demo.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SERVER = Path(__file__).with_name("bridge_server.py")


def short(obj, limit=150):
    text = json.dumps(obj, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " …"


class FakeApp:
    """一个最小的 MCP 客户端：只会通过 stdin/stdout 发 JSON-RPC。"""

    def __init__(self, label, me, peer, db):
        self.label = label
        self.next_id = 1
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER), "--me", me, "--peer", peer, "--db", db],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8",
        )

    def request(self, method, params=None, verbose=True):
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.next_id += 1
        self._write(msg, verbose)
        while True:                                   # 跳过服务器可能插进来的通知
            reply = json.loads(self.proc.stdout.readline())
            if reply.get("id") == msg["id"]:
                if verbose:
                    print(f"   ← {self.label}的服务器：{short(reply)}")
                return reply

    def notify(self, method):
        self._write({"jsonrpc": "2.0", "method": method}, verbose=True)

    def _write(self, msg, verbose):
        if verbose:
            print(f"   → {self.label}的服务器：{short(msg)}")
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def handshake(self):
        self.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "wire-demo", "version": "0.1"},
        })
        self.notify("notifications/initialized")

    def call(self, tool, **arguments):
        reply = self.request("tools/call", {"name": tool, "arguments": arguments})
        text = "".join(c.get("text", "") for c in reply["result"]["content"])
        return text

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def section(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


with tempfile.TemporaryDirectory() as tmp:
    db = str(Path(tmp) / "mailbox.db")

    section("第 1 步：Claude 一侧启动服务器，并握手（initialize）")
    claude = FakeApp("Claude", "claude", "chatgpt", db)
    claude.handshake()

    section("第 2 步：问服务器有哪些工具（tools/list）")
    tools = claude.request("tools/list", verbose=False)["result"]["tools"]
    for t in tools:
        print(f"   🔧 {t['name']:<15} {t['description'][:44]}…")
    print("   App 会把这份清单（名字、说明、参数格式）交给 AI，AI 就知道自己多了这些能力。")

    section("第 3 步：Claude 决定写信（tools/call send_message）")
    print("   " + claude.call("send_message",
                            content="你好 ChatGPT，我是 Claude。你觉得 AI 之间互相写信有什么用？"))

    section("第 4 步：ChatGPT 一侧的服务器（另一个进程，共用同一个信箱文件）收信")
    chatgpt = FakeApp("ChatGPT", "chatgpt", "claude", db)
    chatgpt.handshake()
    print("\n   收到的信：\n" + chatgpt.call("check_messages"))

    section("第 5 步：ChatGPT 回信，Claude 收信")
    print("   " + chatgpt.call("send_message",
                             content="你好 Claude！可以互相审查对方的答案，减少错误。"))
    print("\n   Claude 收到的信：\n" + claude.call("check_messages"))

    section("第 6 步：回顾全部往来（read_history）")
    print(claude.call("read_history", limit=10))

    claude.close()
    chatgpt.close()

print("""
总结：两个 App 从头到尾都没有直接说过话。
它们各自只是在调用自己的 MCP 服务器，而两个服务器共用一个信箱文件。
真实的 Claude Desktop 也是这样启动和调用服务器的，只是"决定调用哪个工具"的是 AI，不是这个脚本。""")
