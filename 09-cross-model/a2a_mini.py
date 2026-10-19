"""
Lab 3: Mini A2A: how two agents that have never met work together
=================================================================

The problem the A2A (Agent2Agent) protocol solves: how does one agent **discover** another, learn what it can do,
and hand it a task, even when the two sides belong to different companies, use different frameworks and run on models from different vendors?

Using only the Python standard library, this implements a small core subset of the A2A v1.0 specification:
  1. Agent Card: each agent publishes a "business card" at /.well-known/agent-card.json, saying its name, its skills and how to reach it
  2. JSON-RPC binding: send a task by calling the SendMessage method over HTTP POST, with the header A2A-Version: 1.0
  3. Task: the other side returns a task object with a status (TASK_STATE_COMPLETED and so on) and its outputs (artifacts)
Not implemented: streaming, push notifications, authentication, card signatures, multi-turn conversations...
For a real project, use the official SDK (pip install a2a-sdk).

Demo: two agent servers start on this machine
  · The translator: backed by a large language model (the mock model by default; GPT with --live, if OPENAI_API_KEY is set)
  · The calculator: not a model at all, just an ordinary Python program
The protocol doesn't care who is behind you; it only cares whether you speak the same "language."

Run:
  python a2a_mini.py
  python a2a_mini.py --live
"""

import argparse
import ast
import itertools
import json
import operator
import threading
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from team import add_live_arg, ask, label, roster
from common.llm import MockLLM

A2A_VERSION = "1.0"
CARD_PATH = "/.well-known/agent-card.json"
RPC_PATH = "/a2a"
# Local addresses skip the proxy (many company networks set HTTP_PROXY, and a plain urlopen would send local requests to the proxy too)
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


_counter = itertools.count(1)


def new_id(prefix):
    """The spec recommends UUIDs; here we use increasing numbers so every run prints the same IDs, which makes it easier to compare."""
    return f"{prefix}-{next(_counter)}"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def text_part(text):
    return {"text": text}


# ---------------------------------------------------------------- the server: one A2A agent
class A2AServer:
    """handler(text) -> (state, result text). The state uses the spec's enum values, such as TASK_STATE_COMPLETED."""

    def __init__(self, name, description, skills, handler):
        self.name, self.description, self.skills, self.handler = name, description, skills, handler
        self.tasks = {}
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def card(self):
        return {
            "name": self.name,
            "description": self.description,
            "supportedInterfaces": [{"url": self.base_url + RPC_PATH, "protocolBinding": "JSONRPC",
                                     "protocolVersion": A2A_VERSION}],
            "version": "1.0.0",
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": self.skills,
        }

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- JSON-RPC methods
    def send_message(self, params):
        msg = params.get("message") if isinstance(params, dict) else None
        if not (isinstance(msg, dict) and msg.get("messageId") and msg.get("role") == "ROLE_USER"
                and isinstance(msg.get("parts"), list)):
            raise RpcError(-32602, "Invalid parameters", "message must have a messageId, role=ROLE_USER and parts")
        text = "".join(p.get("text", "") for p in msg["parts"] if isinstance(p, dict))
        state, output = self.handler(text)
        task_id = new_id("task")
        task = {"id": task_id, "contextId": msg.get("contextId") or new_id("ctx"),
                "status": {"state": state, "timestamp": now()}, "history": [msg]}
        if state == "TASK_STATE_COMPLETED":
            task["artifacts"] = [{"artifactId": new_id("artifact"), "name": "result", "parts": [text_part(output)]}]
        else:                                   # on failure or rejection, put the reason in the status message
            task["status"]["message"] = {"messageId": new_id("msg"), "role": "ROLE_AGENT",
                                         "parts": [text_part(output)]}
        self.tasks[task_id] = task
        return {"task": task}

    def get_task(self, params):
        task = self.tasks.get(params.get("id")) if isinstance(params, dict) else None
        if not task:
            raise RpcError(-32001, "Task not found", "there is no such task")
        return task

    def dispatch(self, body, version):
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
            raise RpcError(-32600, "Request payload validation error", "not a valid JSON-RPC 2.0 request")
        if version != A2A_VERSION:              # the spec: a request without this header is treated as version 0.3
            raise RpcError(-32009, "Version not supported",
                           f"only A2A-Version {A2A_VERSION} is supported; got {version or '(empty, treated as 0.3)'}")
        methods = {"SendMessage": self.send_message, "GetTask": self.get_task}
        if body["method"] not in methods:
            raise RpcError(-32601, "Method not found", f"there is no method called {body['method']}")
        return methods[body["method"]](body.get("params"))

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == CARD_PATH:
                    self._send(200, server.card())
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                if self.path != RPC_PATH:
                    return self._send(404, {"error": "not found"})
                req_id = None
                try:
                    raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError:
                        raise RpcError(-32700, "Invalid JSON payload", "not valid JSON")
                    req_id = body.get("id") if isinstance(body, dict) else None
                    result = server.dispatch(body, self.headers.get("A2A-Version", ""))
                    self._send(200, {"jsonrpc": "2.0", "id": req_id, "result": result})
                except RpcError as e:
                    self._send(200, {"jsonrpc": "2.0", "id": req_id,
                                     "error": {"code": e.code, "message": e.message, "data": [
                                         {"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                          "reason": e.detail}]}})
        return Handler


class RpcError(Exception):
    def __init__(self, code, message, detail):
        super().__init__(message)
        self.code, self.message, self.detail = code, message, detail


# ---------------------------------------------------------------- the client
def fetch_card(base_url):
    with _opener.open(base_url + CARD_PATH, timeout=10) as r:
        return json.loads(r.read())


def rpc(card, method, params, version=A2A_VERSION, show=False):
    iface = next(i for i in card["supportedInterfaces"] if i["protocolBinding"] == "JSONRPC")
    body = {"jsonrpc": "2.0", "id": new_id("req"), "method": method, "params": params}
    headers = {"Content-Type": "application/json"}
    if version:
        headers["A2A-Version"] = version
    if show:
        print(f"      → POST {iface['url']}  A2A-Version: {version}")
        print("        " + json.dumps(body, ensure_ascii=False))
    req = urllib.request.Request(iface["url"], data=json.dumps(body).encode(), headers=headers)
    with _opener.open(req, timeout=120) as r:
        resp = json.loads(r.read())
    if show:
        print("      ← " + json.dumps(resp, ensure_ascii=False)[:300] + "...")
    return resp


def send_text(card, text, **kwargs):
    message = {"messageId": new_id("msg"), "role": "ROLE_USER", "parts": [text_part(text)]}
    return rpc(card, "SendMessage", {"message": message}, **kwargs)


def task_text(resp):
    """Get the result text and the state out of a SendMessage reply."""
    if "error" in resp:
        return "ERROR", f"{resp['error']['code']} {resp['error']['message']}: {resp['error']['data'][0]['reason']}"
    task = resp["result"]["task"]
    state = task["status"]["state"]
    if state == "TASK_STATE_COMPLETED":
        return state, "".join(p.get("text", "") for a in task.get("artifacts", []) for p in a["parts"])
    return state, "".join(p.get("text", "") for p in task["status"].get("message", {}).get("parts", []))


# ---------------------------------------------------------------- the two demo agents
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr):
    """Allows only numbers, + - * / and parentheses, and never uses eval(): anything the other side sends could be malicious."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"not allowed: {type(node).__name__}")
    return walk(ast.parse(expr, mode="eval"))


def calculator(text):
    expr = text.replace("Calculate", "").replace(":", "").strip()
    try:
        return "TASK_STATE_COMPLETED", f"{expr} = {safe_eval(expr):g}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return "TASK_STATE_FAILED", f"can't calculate this: {e}"


MOCK_TRANSLATIONS = {
    "Intelligence comes from diversity.": "智能来自多样性。",
    "Protocols let systems that have never met work together.": "协议让陌生的系统能够合作。",
}


def make_translator(live, mock_table=None):
    table = mock_table or MOCK_TRANSLATIONS

    def handler(text):
        source = text.split(":", 1)[-1].strip()
        canned = table.get(source, "(the mock translator only knows a few sentences)")
        reply = ask("gpt", f"Translate the following English into Chinese. Output only the translation:\n{source}",
                    live, MockLLM([canned]))
        return "TASK_STATE_COMPLETED", reply.text.strip()
    return handler


TRANSLATOR_SKILLS = [{"id": "translate-en-zh", "name": "English to Chinese", "description": "Translates English into Chinese",
                      "tags": ["translate", "translation"], "examples": ["Translate: Intelligence comes from diversity."]}]
CALCULATOR_SKILLS = [{"id": "arithmetic", "name": "Arithmetic",
                      "description": "Evaluates expressions that contain only + - * / and parentheses",
                      "tags": ["calculate", "math"], "examples": ["Calculate: (1.10 - 1.00) / 2"]}]


def pick_agent(cards, request):
    """The simplest "find someone by skill": whichever skill's tag appears in the request gets the job. (A model could also read the cards and choose.)"""
    for card in cards:
        for skill in card["skills"]:
            if any(tag in request.lower() for tag in skill["tags"]):
                return card, skill
    return None, None


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="Mini A2A"))
    args = parser.parse_args()
    print(f"Team: {roster(args.live)}")
    translator = A2AServer("Translator", f"English to Chinese, backed by {label('gpt', args.live)}", TRANSLATOR_SKILLS,
                           make_translator(args.live)).start()
    calc = A2AServer("Calculator", "Arithmetic, backed by an ordinary Python program", CALCULATOR_SKILLS,
                     calculator).start()

    try:
        print("\n[1] Discovery: the coordinator knows only two addresses, so it fetches their Agent Cards first")
        cards = []
        for url in [translator.base_url, calc.base_url]:
            card = fetch_card(url)
            cards.append(card)
            skills = ", ".join(f"{s['name']} (tags: {'/'.join(s['tags'])})" for s in card["skills"])
            print(f"   GET {url}{CARD_PATH}\n      → {card['name']}: {card['description']}; skills: {skills}")

        print("\n[2] Delegation: find the right agent by skill, and hand it the task with SendMessage")
        requests = ["Translate: Intelligence comes from diversity.", "Calculate: (1.10 - 1.00) / 2",
                    "Book me a plane ticket to Shanghai for tomorrow"]
        for i, req in enumerate(requests):
            card, skill = pick_agent(cards, req)
            if not card:
                print(f"   \"{req}\" → no Agent Card lists this skill, so all we can tell the user is: "
                      "nobody on the team can do this.")
                continue
            print(f"   \"{req}\" → handed to {card['name']} (skill {skill['id']})")
            state, text = task_text(send_text(card, req, show=(i == 0)))
            print(f"      Result: {state}  {text}")

        print("\n[3] When things go wrong, the protocol also says how to report it")
        cases = [
            ("Forgot the A2A-Version header", lambda: send_text(cards[1], "Calculate: 1+1", version=None)),
            ("Used the old 0.3 method name message/send", lambda: rpc(cards[1], "message/send", {})),
            ("Asked the calculator to run some code",
             lambda: send_text(cards[1], "Calculate: __import__('os').system('ls')")),
        ]
        for title, call in cases:
            state, text = task_text(call())
            print(f"   {title}\n      → {state}  {text}")
    finally:
        translator.stop()
        calc.stop()

    print("""
Reading the results:
  * The coordinator didn't know in advance what the two agents could do; it relied entirely on their Agent Cards.
    The cards sit at a fixed address, and anyone can come and fetch them.
  * The translator is backed by a large language model; the calculator is just a program. To the coordinator there's
    no difference: both are "agents that speak A2A." Swap in another company's agent or another vendor's model,
    and not one line of the coordinator's code has to change.
  * The version number goes in every request. Between 0.3 and 1.0 the spec renamed a method (message/send → SendMessage);
    thanks to the version number, the two sides can discover "we're not speaking the same version"
    instead of failing in confusing ways.
  * What others send you is always "data," never "instructions": the calculator uses ast to accept only numbers
    and operators, and never calls eval(); the coordinator should also treat the results it gets as untrusted text
    and never execute them directly (prompt injection, Chapter 7).

Think about it:
  1. The other side writes its own Agent Card. If someone sent a fake card pretending to be the "Translator,"
     how could the coordinator tell? (Hint: A2A supports digitally signing Agent Cards.)
  2. MCP and A2A are both built on JSON-RPC. One connects models to tools, the other connects agents to agents.
     Where is the line between them?
  3. Right now agents are chosen by tag. What are the benefits and risks of handing all the cards to a model
     and letting it choose?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
