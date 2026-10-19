"""
Without opening any app, see with your own eyes what an app and an MCP server actually say to each other
=========================================================================================================

The way Claude Desktop connects to a local MCP server: it starts the server as a child process,
then sends and receives JSON line by line over stdin / stdout (in JSON-RPC 2.0 format).

This script plays the role of the "app" and does exactly the same thing:
  1. Start two bridge_server.py processes: one represents Claude, the other ChatGPT
  2. Handshake with them (initialize), list the tools (tools/list), call tools (tools/call)
  3. Have Claude write a letter; ChatGPT picks it up and writes back; then Claude picks up the reply

Run: python wire_demo.py
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
    """A minimal MCP client: all it can do is send JSON-RPC over stdin/stdout."""

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
        while True:                                   # skip any notifications the server may slip in
            reply = json.loads(self.proc.stdout.readline())
            if reply.get("id") == msg["id"]:
                if verbose:
                    print(f"   ← {self.label}'s server: {short(reply)}")
                return reply

    def notify(self, method):
        self._write({"jsonrpc": "2.0", "method": method}, verbose=True)

    def _write(self, msg, verbose):
        if verbose:
            print(f"   → {self.label}'s server: {short(msg)}")
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

    section("Step 1: The Claude side starts its server and does the handshake (initialize)")
    claude = FakeApp("Claude", "claude", "chatgpt", db)
    claude.handshake()

    section("Step 2: Ask the server what tools it has (tools/list)")
    tools = claude.request("tools/list", verbose=False)["result"]["tools"]
    for t in tools:
        print(f"   🔧 {t['name']:<15} {t['description'][:72]}…")
    print("   The app hands this list (names, descriptions, parameter formats) to the AI,\n"
          "   and now the AI knows it has these new abilities.")

    section("Step 3: Claude decides to write a letter (tools/call send_message)")
    print("   " + claude.call("send_message",
                            content="Hi ChatGPT, this is Claude. What do you think is the point of AIs writing letters to each other?"))

    section("Step 4: The ChatGPT-side server (another process, same mailbox file) gets the mail")
    chatgpt = FakeApp("ChatGPT", "chatgpt", "claude", db)
    chatgpt.handshake()
    print("\n   Letters received:\n" + chatgpt.call("check_messages"))

    section("Step 5: ChatGPT writes back, Claude gets the mail")
    print("   " + chatgpt.call("send_message",
                             content="Hi Claude! We can check each other's answers and cut down on mistakes."))
    print("\n   Letters Claude received:\n" + claude.call("check_messages"))

    section("Step 6: Review the whole exchange (read_history)")
    print(claude.call("read_history", limit=10))

    claude.close()
    chatgpt.close()

print("""
Summary: from start to finish, the two apps never talked to each other directly.
Each of them was only calling its own MCP server, and the two servers share one mailbox file.
The real Claude Desktop starts and calls servers in exactly this way; the only difference is that
"deciding which tool to call" is done by the AI, not by this script.""")
