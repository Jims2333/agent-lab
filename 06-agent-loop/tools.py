"""
实验 2：给模型准备"工具"
==========================

模型只会输出文字。要让它"做事"，我们得提供工具，并用一份**说明书**告诉它每个工具能干什么、要传什么参数。
这份说明书用 JSON Schema 写，三家 API（Claude、GPT、DeepSeek）都认这个格式。

模型从来不会自己执行工具。它只是说"我想调用 calculator，参数是 {...}"，
**真正执行的是我们的程序**，执行完再把结果告诉它。

这里定义四个工具：
  calculator   安全地计算算术表达式
  search_repo  在本仓库的文档和代码里搜索关键词
  read_file    读取仓库里某个文件的几行
  list_dir     列出仓库里某个目录下的文件

运行：python tools.py （打印模型看到的工具说明书，并演示几次调用，包括出错的情况）
其他实验会 import 这里的 TOOLS 和 run_tool。
"""

import ast
import json
import operator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # 工具只能在本仓库里活动
TEXT_SUFFIXES = {".md", ".py", ".txt", ".csv"}

TOOLS = [
    {
        "name": "calculator",
        "description": "计算一个算术表达式，支持 + - * / ** 和括号。例如 '26.2 - 15.3'。",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "要计算的算术表达式"}},
            "required": ["expression"],
        },
    },
    {
        "name": "search_repo",
        "description": "在 agent-lab 仓库的所有文档和代码里搜索一个关键词，返回匹配的行（文件路径:行号: 内容），最多 8 条。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要搜索的关键词，越具体越好"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "读取仓库里一个文件的若干行。路径相对于仓库根目录，例如 '02-linear-models/README.md'。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对于仓库根目录的文件路径"},
                "start": {"type": "integer", "description": "从第几行开始（从 1 开始），默认 1"},
                "end": {"type": "integer", "description": "读到第几行（含），默认 start + 39"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "列出仓库里一个目录下的文件和子目录。路径相对于仓库根目录，'.' 表示根目录。",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "相对于仓库根目录的目录路径"}},
            "required": ["path"],
        },
    },
]


class ToolError(Exception):
    """工具执行失败。错误信息会原样告诉模型，让它有机会改正。"""


# ---------------------------------------------------------------- 工具的实现
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression):
    """只允许数字和四则运算。不能用 eval：模型给的字符串可能包含任意代码。"""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ToolError(f"不支持的表达式成分：{ast.dump(node)[:40]}")
    try:
        value = ev(ast.parse(expression, mode="eval").body)
    except SyntaxError:
        raise ToolError(f"表达式写错了：{expression!r}")
    except ZeroDivisionError:
        raise ToolError("除数不能为 0")
    return str(round(value, 10))


def _resolve(path):
    """把模型给的相对路径转成真实路径，并确保它还在仓库里面（防止 ../../ 跑出去）。"""
    target = (REPO / path).resolve()
    if target != REPO and REPO not in target.parents:
        raise ToolError(f"不允许访问仓库以外的路径：{path}")
    return target


def search_repo(query):
    if not query.strip():
        raise ToolError("关键词不能为空")
    hits = []
    for f in sorted(REPO.rglob("*")):
        if f.suffix not in TEXT_SUFFIXES or ".git" in f.parts or "data" in f.parts:
            continue
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if query in line:
                hits.append(f"{f.relative_to(REPO)}:{i}: {line.strip()[:120]}")
                if len(hits) >= 8:
                    return "\n".join(hits)
    return "\n".join(hits) if hits else f"没有找到包含 {query!r} 的内容"


def read_file(path, start=1, end=None):
    f = _resolve(path)
    if not f.is_file():
        raise ToolError(f"文件不存在：{path}（可以先用 list_dir 或 search_repo 找一找）")
    lines = f.read_text(errors="ignore").splitlines()
    start = max(1, int(start))
    end = min(len(lines), int(end) if end else start + 39)
    body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return f"{path}（共 {len(lines)} 行，显示第 {start}～{end} 行）\n{body}"


def list_dir(path):
    d = _resolve(path)
    if not d.is_dir():
        raise ToolError(f"目录不存在：{path}")
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir()
                   if not p.name.startswith(".") and p.name != "__pycache__")
    return "\n".join(names)


IMPLEMENTATIONS = {"calculator": calculator, "search_repo": search_repo,
                   "read_file": read_file, "list_dir": list_dir}


def check_args(tool, args):
    """按 JSON Schema 做最基本的检查：必填参数在不在、类型对不对。"""
    schema = tool["parameters"]
    for name in schema.get("required", []):
        if name not in args:
            raise ToolError(f"缺少必填参数：{name}")
    types = {"string": str, "integer": int, "number": (int, float)}
    for name, value in args.items():
        spec = schema["properties"].get(name)
        if spec is None:
            raise ToolError(f"未知参数：{name}")
        if not isinstance(value, types[spec["type"]]):
            raise ToolError(f"参数 {name} 应该是 {spec['type']}，收到的是 {type(value).__name__}")


def run_tool(name, args, tools=TOOLS, implementations=IMPLEMENTATIONS):
    """执行一个工具调用。返回 (结果文字, 是否出错)。出错不抛异常，而是把错误告诉模型。"""
    tool = next((t for t in tools if t["name"] == name), None)
    if tool is None:
        return f"没有叫 {name} 的工具。可用的工具：{', '.join(t['name'] for t in tools)}", True
    try:
        check_args(tool, args)
        return implementations[name](**args), False
    except ToolError as e:
        return f"出错了：{e}", True


if __name__ == "__main__":
    print("【1】模型看到的工具说明书（JSON Schema）：")
    print(json.dumps(TOOLS[0], ensure_ascii=False, indent=2))
    print(f"   ……一共 {len(TOOLS)} 个工具：{', '.join(t['name'] for t in TOOLS)}\n")

    print("【2】我们的程序替模型执行工具：")
    demos = [
        ("calculator", {"expression": "26.2 - 15.3"}),
        ("calculator", {"expression": "__import__('os').system('rm -rf /')"}),
        ("search_repo", {"query": "Les 3 Brasseurs"}),
        ("read_file", {"path": "../../etc/passwd"}),
        ("read_file", {"path": "02-linear-models/README.md", "start": 1, "end": 3}),
        ("list_dir", {"path": "06-agent-loop"}),
        ("calculator", {"expr": "1+1"}),
        ("send_email", {"to": "boss"}),
    ]
    for name, args in demos:
        result, is_error = run_tool(name, args)
        flag = "❌" if is_error else "✅"
        first = result.splitlines()[0] if result else ""
        more = f"（共 {len(result.splitlines())} 行）" if len(result.splitlines()) > 1 else ""
        print(f"   {flag} {name}({json.dumps(args, ensure_ascii=False)})")
        print(f"      → {first[:90]}{more}")

    print("""
   注意那几个 ❌：
   * 计算器不用 eval，所以"表达式"里藏的代码不会被执行；
   * read_file 不允许用 ../ 跑出仓库；
   * 参数名写错、调用不存在的工具，都会返回一条清楚的错误信息，而不是让程序崩溃。
   模型看到错误信息后，通常能自己改正。这些"防护"是第七章 harness 的雏形。

想一想：
  1. 工具的 description 写得含糊（比如只写"搜索"），模型会犯什么错？
  2. 为什么工具出错时，要把错误信息返回给模型，而不是直接抛异常结束程序？
  3. 如果给模型一个能执行任意 shell 命令的工具，需要加哪些限制？""")
