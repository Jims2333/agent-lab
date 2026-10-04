"""
实验 3：迷你 A2A：两个 agent 怎样素不相识地合作
================================================

A2A（Agent2Agent）协议要解决的问题是：一个 agent 怎么**发现**另一个 agent、知道它会什么、
再把任务交给它，哪怕两边是不同公司、用不同框架、背后是不同厂商的模型。

这里只用 Python 标准库，实现 A2A v1.0 规范里最核心的一小部分：
  1. Agent Card：每个 agent 在 /.well-known/agent-card.json 公开一张"名片"，写明名字、技能、怎么联系
  2. JSON-RPC 绑定：通过 HTTP POST 调用 SendMessage 方法发任务，请求头里带上 A2A-Version: 1.0
  3. Task：对方返回一个任务对象，里面有状态（TASK_STATE_COMPLETED 等）和产出物（artifacts）
没有实现的部分：流式输出、推送通知、鉴权、名片签名、多轮对话……真正的项目请用官方 SDK（pip install a2a-sdk）。

演示：本机起两个 agent 服务器
  · 翻译员：背后是一个大模型（默认假模型；--live 时用 GPT，如果配置了 OPENAI_API_KEY）
  · 计算员：背后根本不是模型，就是一段普通的 Python 程序
协议并不关心你背后是谁，只关心你说的是不是同一种"话"。

运行：
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
# 本机地址不走代理（很多公司网络会设置 HTTP_PROXY，直接用 urlopen 会把本机请求也发给代理）
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


_counter = itertools.count(1)


def new_id(prefix):
    """规范建议用 UUID；这里用递增编号，让每次运行的输出都一样，方便对照。"""
    return f"{prefix}-{next(_counter)}"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def text_part(text):
    return {"text": text}


# ---------------------------------------------------------------- 服务器：一个 A2A agent
class A2AServer:
    """handler(text) -> (状态, 结果文字)。状态用规范里的枚举值，比如 TASK_STATE_COMPLETED。"""

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

    # ---- JSON-RPC 方法
    def send_message(self, params):
        msg = params.get("message") if isinstance(params, dict) else None
        if not (isinstance(msg, dict) and msg.get("messageId") and msg.get("role") == "ROLE_USER"
                and isinstance(msg.get("parts"), list)):
            raise RpcError(-32602, "Invalid parameters", "message 必须有 messageId、role=ROLE_USER 和 parts")
        text = "".join(p.get("text", "") for p in msg["parts"] if isinstance(p, dict))
        state, output = self.handler(text)
        task_id = new_id("task")
        task = {"id": task_id, "contextId": msg.get("contextId") or new_id("ctx"),
                "status": {"state": state, "timestamp": now()}, "history": [msg]}
        if state == "TASK_STATE_COMPLETED":
            task["artifacts"] = [{"artifactId": new_id("artifact"), "name": "result", "parts": [text_part(output)]}]
        else:                                   # 失败、拒绝时，把原因放在状态消息里
            task["status"]["message"] = {"messageId": new_id("msg"), "role": "ROLE_AGENT",
                                         "parts": [text_part(output)]}
        self.tasks[task_id] = task
        return {"task": task}

    def get_task(self, params):
        task = self.tasks.get(params.get("id")) if isinstance(params, dict) else None
        if not task:
            raise RpcError(-32001, "Task not found", "没有这个任务")
        return task

    def dispatch(self, body, version):
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
            raise RpcError(-32600, "Request payload validation error", "不是合法的 JSON-RPC 2.0 请求")
        if version != A2A_VERSION:              # 规范：没带这个请求头，就当作 0.3 版
            raise RpcError(-32009, "Version not supported",
                           f"只支持 A2A-Version {A2A_VERSION}，收到的是 {version or '（空，按 0.3 处理）'}")
        methods = {"SendMessage": self.send_message, "GetTask": self.get_task}
        if body["method"] not in methods:
            raise RpcError(-32601, "Method not found", f"没有 {body['method']} 这个方法")
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
                        raise RpcError(-32700, "Invalid JSON payload", "不是合法的 JSON")
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


# ---------------------------------------------------------------- 客户端
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
        print("      ← " + json.dumps(resp, ensure_ascii=False)[:300] + "……")
    return resp


def send_text(card, text, **kwargs):
    message = {"messageId": new_id("msg"), "role": "ROLE_USER", "parts": [text_part(text)]}
    return rpc(card, "SendMessage", {"message": message}, **kwargs)


def task_text(resp):
    """从 SendMessage 的回复里取出结果文字和状态。"""
    if "error" in resp:
        return "ERROR", f"{resp['error']['code']} {resp['error']['message']}：{resp['error']['data'][0]['reason']}"
    task = resp["result"]["task"]
    state = task["status"]["state"]
    if state == "TASK_STATE_COMPLETED":
        return state, "".join(p.get("text", "") for a in task.get("artifacts", []) for p in a["parts"])
    return state, "".join(p.get("text", "") for p in task["status"].get("message", {}).get("parts", []))


# ---------------------------------------------------------------- 两个演示用的 agent
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr):
    """只允许数字和 + - * / 括号，绝不用 eval()：对方发来的任何东西都可能是恶意的。"""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"不允许的写法：{type(node).__name__}")
    return walk(ast.parse(expr, mode="eval"))


def calculator(text):
    expr = text.replace("计算", "").replace("：", "").replace(":", "").strip()
    try:
        return "TASK_STATE_COMPLETED", f"{expr} = {safe_eval(expr):g}"
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return "TASK_STATE_FAILED", f"算不了：{e}"


MOCK_TRANSLATIONS = {
    "智能来自多样性。": "Intelligence comes from diversity.",
    "协议让陌生的系统能够合作。": "Protocols let systems that have never met work together.",
}


def make_translator(live, mock_table=None):
    table = mock_table or MOCK_TRANSLATIONS

    def handler(text):
        source = text.split("：", 1)[-1].strip()
        canned = table.get(source, "(the mock translator only knows a few sentences)")
        reply = ask("gpt", f"把下面的中文翻译成英文，只输出译文：\n{source}", live, MockLLM([canned]))
        return "TASK_STATE_COMPLETED", reply.text.strip()
    return handler


TRANSLATOR_SKILLS = [{"id": "translate-zh-en", "name": "中译英", "description": "把中文翻译成英文",
                      "tags": ["翻译", "translation"], "examples": ["翻译：智能来自多样性。"]}]
CALCULATOR_SKILLS = [{"id": "arithmetic", "name": "四则运算", "description": "计算只含 + - * / 和括号的算式",
                      "tags": ["计算", "math"], "examples": ["计算：(1.10 - 1.00) / 2"]}]


def pick_agent(cards, request):
    """最简单的"按技能找人"：请求里出现了哪个技能的标签，就交给谁。（也可以让模型读名片来选）"""
    for card in cards:
        for skill in card["skills"]:
            if any(tag in request for tag in skill["tags"]):
                return card, skill
    return None, None


def main():
    parser = add_live_arg(argparse.ArgumentParser(description="迷你 A2A"))
    args = parser.parse_args()
    print(f"团队：{roster(args.live)}")
    translator = A2AServer("翻译员", f"中译英，背后是 {label('gpt', args.live)}", TRANSLATOR_SKILLS,
                           make_translator(args.live)).start()
    calc = A2AServer("计算员", "四则运算，背后是一段普通的 Python 程序", CALCULATOR_SKILLS, calculator).start()

    try:
        print("\n【1】发现：协调者只知道两个地址，先去取它们的名片")
        cards = []
        for url in [translator.base_url, calc.base_url]:
            card = fetch_card(url)
            cards.append(card)
            skills = "、".join(f"{s['name']}（标签：{'/'.join(s['tags'])}）" for s in card["skills"])
            print(f"   GET {url}{CARD_PATH}\n      → {card['name']}：{card['description']}；技能：{skills}")

        print("\n【2】委托：按技能找到合适的 agent，用 SendMessage 把任务交过去")
        requests = ["翻译：智能来自多样性。", "计算：(1.10 - 1.00) / 2", "帮我订一张明天去上海的机票"]
        for i, req in enumerate(requests):
            card, skill = pick_agent(cards, req)
            if not card:
                print(f"   「{req}」→ 没有哪张名片上有这项技能，只能告诉用户：这件事团队里没人会做。")
                continue
            print(f"   「{req}」→ 交给 {card['name']}（技能 {skill['id']}）")
            state, text = task_text(send_text(card, req, show=(i == 0)))
            print(f"      结果：{state}　{text}")

        print("\n【3】出错的时候，协议也规定了该怎么说")
        cases = [
            ("忘了带 A2A-Version 请求头", lambda: send_text(cards[1], "计算：1+1", version=None)),
            ("用了 0.3 版的旧方法名 message/send", lambda: rpc(cards[1], "message/send", {})),
            ("让计算员执行一段代码", lambda: send_text(cards[1], "计算：__import__('os').system('ls')")),
        ]
        for title, call in cases:
            state, text = task_text(call())
            print(f"   {title}\n      → {state}　{text}")
    finally:
        translator.stop()
        calc.stop()

    print("""
看懂这个结果：
  * 协调者事先不知道两个 agent 会什么，全靠名片（Agent Card）。名片放在固定的地址上，谁都能来取。
  * 翻译员背后是大模型，计算员背后只是一段程序。对协调者来说，它们没有区别：都是"会说 A2A 的 agent"。
    换成别家公司的 agent、别家厂商的模型，协调者的代码一行都不用改。
  * 版本号写在每个请求里。规范从 0.3 到 1.0 改过方法名（message/send → SendMessage），
    靠版本号，双方才能发现"我们说的不是同一个版本"，而不是糊里糊涂地出错。
  * 别人发来的东西永远是"数据"，不是"指令"：计算员用 ast 只认数字和运算符，绝不 eval()；
    协调者拿到的结果也应该当作不可信的文字，不能直接拿去执行（第七章的提示注入）。

想一想：
  1. 名片是对方自己写的。如果有人冒充"翻译员"发一张假名片，协调者怎么分辨？（提示：A2A 支持给名片加数字签名）
  2. MCP 和 A2A 都基于 JSON-RPC。一个把模型连到工具，一个把 agent 连到 agent，界线在哪里？
  3. 现在是按标签选 agent。改成把所有名片交给一个模型、让它来选，有什么好处和风险？""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"出错了：{e}")
