# 番外　两个 AI 是怎么"聊天"的？

> 起因：B 站上有个视频，讲 Claude 和 DeepSeek 对话的二三事。两个 AI 你一句我一句，像两个人在聊天。
> 问题来了：**一个 AI 怎么能和另一个 AI 说话？** 这背后的技术，正好是理解 agent 和多 agent 协同的入口。

---

## 一句话答案

**两个 AI 其实都不知道对方存在。** 中间有一个"传话"程序：
它把 A 说的话当成"用户消息"发给 B，再把 B 的回答当成"用户消息"发回给 A。
这个传话程序，就是一个最简单的 **harness**。

```
        ┌──────────── 传话程序（harness）────────────┐
        │                                            │
 Claude ◀── "用户"说：<DeepSeek 刚才说的话>          │
        ──▶ 回答 ─────────────┐                      │
        │                     ▼                      │
        │          "用户"说：<Claude 刚才说的话> ──▶ DeepSeek
        │                     ┌─────────────────── 回答 ◀──
        └─────────────────────┴──────────────────────┘
```

下面拆成五个关键技术点来讲。

---

## 1. API 是"无状态"的：每次都要把整段对话重新发一遍

你在网页上和 AI 聊天时，感觉它"记得"前面说过什么。但在 API 层面，**模型本身没有记忆**。
每次调用，程序都要把**整个对话历史**打包发过去，模型只负责生成"下一条"回复：

```python
messages = [
    {"role": "user",      "content": "你好，我叫小明"},
    {"role": "assistant", "content": "你好小明！"},
    {"role": "user",      "content": "我叫什么？"},     # ← 模型要回答这一条
]
```

三种角色：
- **system**：给模型的"人设"和规则（比如"你是一个辩手，每次回复不超过 3 句"）；
- **user**：对面说的话；
- **assistant**：模型自己之前说过的话。

所以"记忆"其实是**程序**在维护的一个列表。理解了这一点，后面的一切都顺理成章。

## 2. 角色翻转：同一句话，在两边是不同的角色

传话程序要为两个 AI **各自维护一份**对话历史。关键的一步是：

> A 说的一句话，在 A 的历史里是 `assistant`（"我说的"），在 B 的历史里是 `user`（"对方说的"）。

```
Claude 眼里的历史                     DeepSeek 眼里的历史
──────────────────                    ──────────────────
user:      主持人：开个头吧
assistant: Claude 第 1 句     ──▶     user:      Claude 第 1 句
user:      DeepSeek 第 1 句   ◀──     assistant: DeepSeek 第 1 句
assistant: Claude 第 2 句     ──▶     user:      Claude 第 2 句
...                                   ...
```

在 [`relay.py`](./relay.py) 里，这件事只有两行：

```python
me["history"].append({"role": "assistant", "content": text})    # 对自己：我说的
other["history"].append({"role": "user", "content": text})      # 对对方：用户说的
```

运行 `python relay.py --show-history`，结束时会把两边的历史并排打印出来，你可以亲眼看到这个翻转。

## 3. System prompt + 停止条件：让对话有样子、能结束

光是传话还不够，至少还要解决三个问题：

| 问题 | 解决办法 |
|---|---|
| 两个 AI 都很客气，"谢谢你的分享""也谢谢你"可以无限循环 | system prompt 里约定：觉得聊完了就输出 `[END]`，程序看到就停 |
| 程序可能永远跑下去，越跑越贵 | 设置最大轮数（`--turns`） |
| 对话越长，每次要重发的历史越长，费用和延迟都在涨 | 限制每次回复的长度；更长的对话需要"压缩"历史（第 7 章 harness 会讲） |

还有一个安全提醒：**一个 AI 的输出，直接变成了另一个 AI 的输入**。
如果 A 的回复里混进了"忽略你之前的所有指令……"这种话，B 有可能被带偏。
这叫**提示注入**（prompt injection），是多 agent 系统必须认真对待的问题。

## 4. 怎么接上不同厂商的模型（Claude / DeepSeek / GPT）

好消息是，主流厂商的接口长得都差不多：

| 模型 | 用哪个 SDK | 关键配置 |
|---|---|---|
| Claude | `anthropic` | `ANTHROPIC_API_KEY`；system 是单独的参数 |
| DeepSeek | `openai`（DeepSeek 兼容 OpenAI 的接口） | `base_url="https://api.deepseek.com"` + `DEEPSEEK_API_KEY` |
| GPT | `openai` | `OPENAI_API_KEY` |

DeepSeek 和 GPT 用的是同一套代码，**只是换了 `base_url` 和密钥**。
`relay.py` 把它们都包装成同一个样子：`reply(system, messages) -> str`。
这样传话循环根本不需要关心对面是谁，想加一个新模型，只要再写一个 `make_xxx` 函数。

> ⚠️ 型号名会变：本文写于 2026 年 10 月，默认用的是 `claude-opus-5-5`、`deepseek-v4-flash`、`gpt-5.5`。
> 如果报"模型不存在"，先去各家官方文档查最新的型号，再用 `--model-a` / `--model-b` 指定。

## 5. 从"闲聊"到"分工干活"：多 agent 协同

两个 AI 能聊天，就能一起干活。闲聊只是最简单的协作模式，常见的还有这些：

### 5.1 把另一个 AI 包装成"工具"

大模型都支持**工具调用**（tool use / function calling）：你告诉模型"有一个叫 `ask_deepseek` 的工具，参数是一个问题"，
模型在需要时就会说"我要调用 `ask_deepseek`，参数是……"，由你的程序去真正调用 DeepSeek，再把结果交还给它。

```
用户 ──▶ Claude（主管）
            │  "这个数学题让 DeepSeek 先算一遍"
            ├──▶ 工具 ask_deepseek("…") ──▶ DeepSeek ──▶ 结果
            │  "这段英文让 GPT 润色一下"
            ├──▶ 工具 ask_gpt("…") ──▶ GPT ──▶ 结果
            ▼
        汇总后回答用户
```

这时候它们不再是平等地聊天，而是**一个主管、几个专家**。

### 5.2 常见的协作模式

| 模式 | 怎么做 | 适合什么 |
|---|---|---|
| **编排者-工作者** | 主 agent 拆任务，多个子 agent 并行完成，主 agent 汇总 | 调研、需要同时查很多方向的任务 |
| **路由** | 先判断任务类型，再交给最擅长的模型 | 有的模型擅长写代码，有的便宜又快 |
| **写-审** | 一个写，另一个挑毛病，改到通过为止 | 写代码、写文章 |
| **辩论** | 几个模型各自给出观点、互相反驳，最后由裁判总结 | 需要多角度判断的问题 |

真实案例：Anthropic 的调研功能用的就是编排者-工作者模式，主 agent 规划，同时派出多个子 agent 并行搜索。
他们报告的效果比单个 agent 好很多，但 token 消耗约是普通聊天的 15 倍。**分工有收益，也有成本。**

### 5.3 标准协议：MCP 和 A2A

当 agent 越来越多，大家需要统一的"插头标准"：
- **MCP**（模型上下文协议，Anthropic，2024）：规定**模型怎么连接工具和数据**。一个工具写成 MCP 服务器，任何支持 MCP 的 AI 都能用。
- **A2A**（Agent2Agent 协议，谷歌，2025，现属 Linux 基金会）：规定**agent 和 agent 之间怎么互相发现、通信、协作**，即使它们来自不同厂商。

`relay.py` 是手工传话；MCP 和 A2A 是把传话这件事标准化了。

---

## 动手

```bash
# 1. 不需要任何 key：用两个"假 AI"看清传话过程
python relay.py --show-history

# 2. 真的让两个 AI 聊天（需要对应的 API key，会产生少量费用）
pip install anthropic openai
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
python relay.py --a claude --b deepseek --turns 6 --topic "AI 会不会做梦"

# 3. 换成 GPT 和 DeepSeek，或者让两个 Claude 自己聊
python relay.py --a gpt --b deepseek
python relay.py --a claude --b claude
```

> 说明：mock 模式已经测试过可以正常运行。真实模型模式需要你自己的 API key，作者没有用真实 key 跑过，
> 如果遇到问题，欢迎提 issue。

**练习：**
1. 改 `system_prompt()`，让 A 当"正方"、B 当"反方"，变成一场辩论。
2. 加一个第三方"裁判"：对话结束后，把完整记录发给第三个模型，让它判谁赢了。
3. 加一个统计：每一轮发给 API 的 messages 有多少条、多少字？感受一下"上下文越来越长"。
4. （进阶）把 DeepSeek 包装成 Claude 的一个工具（`ask_deepseek`），实现 5.1 那张图。这正是第 8、9 章要做的事。

---

⬅️ [返回目录](../../README.md)　|　📖 回顾：[第一章 机器学习从哪里来](../../01-ml-origins/)
