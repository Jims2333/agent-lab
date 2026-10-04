# 第九章　跨模型协作：让 Claude、GPT、DeepSeek 一起干活

> 这是最后一章。还记得番外 1 里，Claude 和 DeepSeek 隔着一段 Python 程序聊天吗？
> 那段程序只是"手工传话"。这一章要回答的是：当参与者越来越多、来自不同的公司，
> 彼此素不相识时，它们靠什么合作？答案很朴素：**协议**。
> 这个答案，人类在五十多年前搭建互联网的时候，就已经找到过一次了。

**怎么读这一章：** 先看故事，再跑实验，最后看[概念解释](#概念解释)。实验默认用假模型，不需要 API key。

| 时间 | 发生了什么 | 对应实验 / 概念 |
|---|---|---|
| 1969 年 4 月 | 一个研究生在浴室里写下 RFC 1 | 协议 |
| 1974 年 5 月 | 瑟夫和卡恩：网络的网络 | 适配层 |
| 1989 年 3 月 | 伯纳斯-李的提议："模糊，但令人兴奋" | 地址与发现 |
| 2016 年 6 月 | 语言服务器协议：把 M×N 变成 M+N | 🧪 `router.py` |
| 2024 年 11 月 | MCP：AI 应用的"USB-C 接口" | MCP |
| 2025 年 4 月 | A2A：agent 之间的名片和对话 | 🧪 `a2a_mini.py` |
| 2026 年 3 月 | A2A 1.0 正式版发布 | 🧪 `cross_check.py`、`capstone.py` |

---

## 序幕　1969 年 4 月 7 日：浴室里的"征求意见稿"

1969 年春天，美国国防部资助的 ARPANET 正准备把几所大学的计算机连起来。
线路和交换设备有人负责，可是**两台计算机连上以后该怎么"说话"**，谁也没有规定。
这件事落到了几所大学的一群研究生头上，他们自称"网络工作组"（Network Working Group）。

加州大学洛杉矶分校的研究生**史蒂夫·克罗克（Steve Crocker）** 负责把大家的讨论记下来。
据他 2009 年在《纽约时报》上的回忆，他当时和朋友合住，为了不吵醒熟睡的室友，是在浴室里熬夜写完第一份笔记的。
他还很担心：一群研究生写的东西，会不会显得太自以为是、冒犯到那些有名望的教授？
于是他给笔记起了一个尽量谦虚的名字：**"征求意见稿"（Request for Comments，RFC）**。

1969 年 4 月 7 日，RFC 1《主机软件》（Host Software）寄了出去。
这个谦虚的名字一直沿用到今天：互联网的各种规范仍然以 RFC 的形式发布，编号早已超过九千。
后面你会看到，A2A 规范里"名片放在哪个地址"这样的细节，引用的也是一份 RFC。

> 💡 伏笔：→ [协议](#1-协议)

---

## 第一幕　1974 年 5 月：网络的网络

ARPANET 跑起来以后，新的问题来了：世界上不止一张网。卫星网络、无线电分组网络……
每张网的包大小不同、出错的方式不同、速度也不同。怎样让**不同的网络**互相连通？

1974 年 5 月，**温顿·瑟夫（Vinton Cerf）** 和 **罗伯特·卡恩（Robert Kahn）** 在《IEEE 通信汇刊》上发表了
《一种用于分组网络互联的协议》。他们的思路是：**不要求每张网都一样**，
只要求大家在边界上（网关）遵守同一套规矩，各自内部爱怎么实现就怎么实现。
这套规矩就是 TCP，后来拆成了今天熟悉的 TCP 和 IP 两层。

1983 年 1 月 1 日是 ARPANET 的"切换日"（flag day）：所有连在网上的主机，都要从旧协议换成 TCP/IP。
很多人把这一天看作互联网真正的生日。

这正是第五章那个 `common/llm.py` 在做的事：Claude 和 GPT、DeepSeek 的接口各不相同，
我们不去改它们，只在边界上加一层**适配层**，让上面的程序只跟一种统一的格式打交道。

> 💡 伏笔：→ [适配层与 M×N 问题](#3-适配层与-mn-问题)

---

## 第二幕　1989 年 3 月，CERN：模糊，但令人兴奋

1989 年 3 月，欧洲核子研究中心（CERN）的 **蒂姆·伯纳斯-李（Tim Berners-Lee）** 交给上司 **迈克·森德尔（Mike Sendall）**
一份提议，题目是《信息管理：一个提议》，封面上画满了互相连接的圆圈和方框。
森德尔在上面写了一句后来很有名的批语：**"模糊，但令人兴奋"**（Vague but exciting）。

接下来，伯纳斯-李写出了第一个浏览器和服务器，定下了 HTTP、HTML 和 URL 这几个概念。
其中 URL 最朴素也最关键：**任何资源都有一个地址，任何人拿着地址都能找到它**。
1993 年 4 月 30 日，CERN 宣布放弃这套软件的全部知识产权，任何人都可以使用、修改和分发。
开放，让万维网迅速传遍了世界。

三十多年后，A2A 协议给每个 agent 规定了一个固定的地址来放它的"名片"，用的还是同一个思路：
**先让别人找得到你，合作才有可能开始。**

> 💡 伏笔：→ [A2A 与 Agent Card](#5-a2a-与-agent-card)

---

## 第三幕　2016 年 6 月：把 M×N 变成 M+N

写代码的编辑器有很多（VS Code、Vim、Emacs……），编程语言也有很多（Python、Java、Go……）。
想让每个编辑器都支持每种语言的"跳转到定义""自动补全"，就要写 M×N 个插件。

2016 年 6 月 27 日，在旧金山的 DevNation 大会上，微软宣布和 Red Hat、Codenvy 一起推动 **语言服务器协议（LSP）**：
每种语言只要实现一个"语言服务器"，每个编辑器只要实现一个"客户端"，双方用基于 JSON 的协议通信。
**M×N 个插件，变成了 M+N 个实现。**

八年后，有人把同样的思路搬到了 AI 上。

> 🧪 **动手**：运行 [`router.py`](./router.py)。一个产品同时接了三家模型，路由器按请求类型派单，
> 一家出故障就自动换下一家。这一切都依赖同一个适配层。

> 💡 伏笔：→ [适配层与 M×N 问题](#3-适配层与-mn-问题)、[路由与故障转移](#7-路由故障转移与级联)

---

## 第四幕　2024 年 11 月：AI 的 USB-C 接口

每个 AI 应用都想连上各种工具和数据：文件、数据库、日历、代码仓库……
每个应用给每个工具写一遍对接代码，又是一个 M×N 问题。

2024 年 11 月 25 日，Anthropic 发布了 **模型上下文协议（Model Context Protocol，MCP）**，
由工程师 David Soria Parra 和 Justin Spahr-Summers 设计。它明确借鉴了 LSP：
工具和数据源实现一个"MCP 服务器"，AI 应用实现一个"MCP 客户端"，双方用 JSON-RPC 2.0 通信。
MCP 的官方文档打了一个比方：**MCP 就像 AI 应用的 USB-C 接口。**

真正让 MCP 变成行业标准的，是竞争对手的加入。2025 年 3 月，OpenAI 的首席执行官**山姆·奥特曼（Sam Altman）** 公开表示：
"大家都喜欢 MCP，我们很高兴在我们的产品中加入对它的支持。"
2025 年 12 月 9 日，Anthropic 把 MCP 捐给了 Linux 基金会旗下新成立的**智能体 AI 基金会（Agentic AI Foundation）**，
这个基金会由 Anthropic、Block 和 OpenAI 共同发起。

番外 2 里，让 ChatGPT 和 Claude 两个桌面 App 互相传话的那个"信箱"，就是一个 MCP 服务器。

> 💡 伏笔：→ [MCP](#4-mcp)

---

## 第五幕　2025 年 4 月：agent 之间的名片

MCP 解决的是"模型怎么用工具"。可是，如果对面不是一个工具，而是**另一个 agent** 呢？
它有自己的判断，可能要干很久，可能中途需要你补充信息，背后可能是另一家公司、另一家厂商的模型。

2025 年 4 月 9 日，谷歌在 Google Cloud Next 大会上发布了 **A2A（Agent2Agent）协议**，
首批有五十多家合作伙伴支持。谷歌把它定位为 MCP 的补充：MCP 连接 agent 和工具，A2A 连接 agent 和 agent。
2025 年 6 月 23 日，Linux 基金会宣布成立 A2A 项目，谷歌把协议规范、SDK 和开发工具都移交了过去，
亚马逊云、思科、微软、Salesforce、SAP、ServiceNow 等公司一同加入。

A2A 的核心设计很像一次商务往来：
- 每个 agent 在 `/.well-known/agent-card.json` 这个固定地址放一张 **Agent Card（名片）**，写明自己是谁、会什么、怎么联系；
- 想合作的一方先取名片，再按名片上的地址，用 JSON-RPC 调用 `SendMessage` 把任务发过去；
- 对方返回一个 **Task（任务）**，里面有任务状态和产出物。

2026 年 3 月 12 日，A2A 发布了 1.0 版：方法名统一改成了 `SendMessage` 这样的写法，
客户端必须在每个请求里注明自己用的协议版本。（给名片加数字签名防伪的功能，从 2025 年 7 月的 0.3 版就有了。）

> 🧪 **动手**：运行 [`a2a_mini.py`](./a2a_mini.py)。用标准库实现 A2A 1.0 的一个最小子集，
> 在本机起两个 agent：一个背后是大模型，一个背后只是一段程序。看看"素不相识"的 agent 怎样合作。

> 🧪 **动手**：运行 [`cross_check.py`](./cross_check.py)，看看不同厂商的模型互相核对答案，什么时候有用、什么时候没用。
> 最后运行毕业项目 [`capstone.py`](./capstone.py)：三家模型分工，给这个仓库做一张学习路线卡片。

> 💡 伏笔：→ [A2A 与 Agent Card](#5-a2a-与-agent-card)、[MCP 和 A2A 的分工](#6-mcp-和-a2a-的分工)、
> [模型多样性](#8-模型多样性与错误相关性)、[跨组织协作的安全](#10-跨组织协作的安全)

---

## 尾声：协议让陌生人合作

| 协议 | 年代 | 让谁和谁说上话 | 关键的设计 |
|---|---|---|---|
| RFC 系列 | 1969– | 设计网络的人和人 | 公开讨论、谁都能提意见 |
| TCP/IP | 1974– | 不同的网络 | 只约定边界上的规矩，不管内部怎么实现 |
| HTTP / URL | 1989– | 浏览器和网站 | 每样东西都有地址；开放、免费 |
| LSP | 2016– | 编辑器和编程语言 | M×N 变成 M+N |
| MCP | 2024– | AI 应用和工具、数据 | 模型的"USB-C 接口" |
| A2A | 2025– | agent 和 agent | 名片 + 任务；不管对方背后是谁 |

这些协议没有一个是靠"让大家都变成一样"成功的。它们只规定**边界上怎么说话**，把各自内部的自由留给了各自。
这也是多厂商协作的正确姿势：不必押注哪一家模型最好，而是搭一个能让它们各展所长、互为备份、互相检查的系统。

到这里，你已经从 1943 年的一个人工神经元，一路走到了一支跨厂商的 agent 团队。全书完。

---

## 动手：四个小实验

```bash
python router.py          # 路由与故障转移（默认用假模型）
python cross_check.py     # 跨厂商交叉核对 + 错误相关性的思想实验
python a2a_mini.py        # 迷你 A2A：名片、SendMessage、任务、出错处理
python capstone.py        # 毕业项目：三家模型分工做一张学习路线卡片

# 配置了哪家的 key，哪家就用真实模型；没配置的继续用假模型
export ANTHROPIC_API_KEY=...   # 或 OPENAI_API_KEY、DEEPSEEK_API_KEY，有一两家就行
python capstone.py --live
```

| 实验 | 你应该看到 | 思考题（脚本末尾也有） |
|---|---|---|
| `router.py` | 8 个请求被分到 4 类；gpt 出故障时，推理题自动转给 claude；按路由表分配的费用约是"全部交给 claude"的三分之一 | 怎样用"级联"代替分类：先便宜后昂贵？ |
| `cross_check.py` | 只问一家 4/6，同一家问三次还是 4/6，三家投票 5/6；两家核对挑出 2 道有问题的题，漏掉 1 道"两家都错"的；思想实验里错误越相关，多问几家越没用 | 三家模型的错误相关程度高还是低？怎么测？ |
| `a2a_mini.py` | 取名片 → 按技能派活 → 拿回 Task；订机票没人会；缺版本号报 -32009，旧方法名报 -32601，代码注入被拒 | 有人发一张假名片冒充翻译员，怎么办？ |
| `capstone.py` | 九个调研员并行；偷懒的那个被审稿员打回重做；翻译通过 A2A 完成；最后输出卡片和按厂商分列的账单 | 为什么让便宜的模型当调研员、贵的当审稿员？ |

关于真实模型模式：四个脚本的真实 API 路径我用模拟服务器测试过（确认不会崩溃、格式不对时有兜底），
但没有用真实的 API key 跑过。各家模型的回答会和假模型的剧本不同，比如交叉核对里的陷阱题，现在的模型可能全都答对。
`router.py` 里 GPT-5.5 和 DeepSeek V4 Flash 的价格来自第三方的价格汇总网站，以各家官网为准。

---

## 概念解释

### 1. 协议

一组事先约定好的规矩：消息长什么样、按什么顺序发、出错了怎么说。
有了协议，双方不需要了解对方的内部实现，只要都遵守同一套规矩就能合作。
好的协议往往只规定"边界"，并且公开、免费，谁都可以实现。

📖 登场：[序幕](#序幕1969-年-4-月-7-日浴室里的征求意见稿)

### 2. JSON-RPC

一种很简单的"远程调用"格式：请求是一个 JSON 对象，写明 `jsonrpc: "2.0"`、`id`、`method`（方法名）和 `params`（参数）；
回复里要么有 `result`，要么有 `error`（包含错误码和说明）。
LSP、MCP 和 A2A 的 JSON-RPC 绑定都建立在它上面，所以学会一个，另外几个就很好懂了。

📖 登场：[第三幕](#第三幕2016-年-6-月把-mn-变成-mn)、[第四幕](#第四幕2024-年-11-月ai-的-usb-c-接口)　🧪 `a2a_mini.py`

### 3. 适配层与 M×N 问题

M 个使用方要对接 N 个提供方，如果两两对接，就要写 M×N 份代码。
定一个统一的接口，每个使用方和提供方各实现一次，就只需要 M+N 份。
本仓库的 `common/llm.py` 就是一个小小的适配层：上面的程序只认一种"中立格式"，
由它负责翻译成 Claude、GPT、DeepSeek 各自的格式（system 放哪里、工具调用怎么表示、工具结果怎么回填）。

📖 登场：[第一幕](#第一幕1974-年-5-月网络的网络)、[第三幕](#第三幕2016-年-6-月把-mn-变成-mn)　🧪 `router.py`

### 4. MCP

模型上下文协议。三种角色：**宿主**（Host，比如 Claude Desktop、ChatGPT、IDE）、
宿主里的**客户端**（Client，和一个服务器保持一对一的连接）、提供能力的**服务器**（Server）。
服务器可以提供工具（tools）、资源（resources）和提示模板（prompts）。
写一次 MCP 服务器，所有支持 MCP 的 AI 应用都能用。番外 2 的信箱就是一个例子。

📖 登场：[第四幕](#第四幕2024-年-11-月ai-的-usb-c-接口)

### 5. A2A 与 Agent Card

A2A 规定 agent 之间怎么发现彼此、怎么委托任务：
- **Agent Card**：放在 `/.well-known/agent-card.json` 的名片，包括名字、描述、技能（skills）、联系方式（supportedInterfaces）、支持的能力；
- **Message**：一条消息，由若干 **Part** 组成（文字、文件、结构化数据），角色是 `ROLE_USER` 或 `ROLE_AGENT`；
- **Task**：一次委托，有状态（`TASK_STATE_WORKING`、`TASK_STATE_COMPLETED`、`TASK_STATE_FAILED`、`TASK_STATE_INPUT_REQUIRED` 等）和产出物（artifacts）；
- **绑定**：同一套语义可以走 JSON-RPC、gRPC 或 HTTP+JSON 三种方式传输。

📖 登场：[第二幕](#第二幕1989-年-3-月cern模糊但令人兴奋)、[第五幕](#第五幕2025-年-4-月agent-之间的名片)　🧪 `a2a_mini.py`

### 6. MCP 和 A2A 的分工

一个粗略但好用的区分：**MCP 连接 agent 和工具**，对面是"被调用的能力"，调用完就结束；
**A2A 连接 agent 和 agent**，对面是"有自主性的同事"，任务可能要做很久、中途要补充信息、背后是谁你不知道也不需要知道。
一个 agent 完全可以同时用这两种协议：对内用 MCP 调工具，对外用 A2A 跟别的 agent 合作。

📖 登场：[第五幕](#第五幕2025-年-4-月agent-之间的名片)

### 7. 路由、故障转移与级联

**路由**：先判断请求的类型，再交给合适的模型。**故障转移**：首选的那家出错（限流、宕机），自动换备选。
**级联**（cascade）：先让便宜的模型回答，检查不过关再交给更贵的模型。
2023 年斯坦福的 FrugalGPT 论文报告，这类组合策略在他们的实验里能用低得多的成本达到和最强单个模型相当的效果。
不管用哪种，前提都是你有一套评测，知道"哪类任务交给谁就够了"。

📖 登场：[第三幕](#第三幕2016-年-6-月把-mn-变成-mn)　🧪 `router.py`

### 8. 模型多样性与错误相关性

多问几个模型、取多数或者互相核对，能不能减少错误，取决于它们的错误是不是"相关"的：
同一个模型问很多遍，错误几乎完全相关，多问没用；不同厂商的模型训练方式不同，错误可能没那么相关，多问才有用。
"两家一致就放行、不一致交给人"这种做法不需要知道标准答案，代价是"两家都错且错得一样"的情况会漏过去。

📖 登场：[第五幕](#第五幕2025-年-4-月agent-之间的名片)　🧪 `cross_check.py`

### 9. 版本协商

协议会升级，旧的客户端和新的服务器要能发现"我们说的不是同一个版本"。
A2A 要求客户端在每个请求里带上 `A2A-Version` 请求头；服务器不支持这个版本，就返回明确的错误（`VersionNotSupportedError`，JSON-RPC 错误码 -32009），
而不是糊里糊涂地按错误的格式处理。

📖 登场：[第五幕](#第五幕2025-年-4-月agent-之间的名片)　🧪 `a2a_mini.py`

### 10. 跨组织协作的安全

和别人的 agent 合作，意味着信任边界变宽了：
- **名片可能是假的**：A2A 支持给名片加数字签名，用来验证名片的来源；
- **对方发来的都是数据，不是指令**：既不能 `eval()` 对方发来的内容，也不能让对方的回复改变自己的行为（提示注入，第七章）；
- **最小权限**：只把完成任务所需的信息交给对方，不要把整个上下文、更不要把 API key 发过去；
- **key 只放在环境变量里**，不写进代码，也不提交到仓库。

📖 登场：[第五幕](#第五幕2025-年-4-月agent-之间的名片)　🧪 `a2a_mini.py`

### 11. 全景图

把整本书串起来：模型本身是一台"猜下一个字"的机器（第一～五章）；
给它工具和循环，它就成了 agent（第六章）；
套上 harness，它才能可靠地干活（第七章）；
多个 agent 分工协作，能干一个 agent 干不完的活（第八章）；
有了适配层和协议，不同厂商、不同公司的 agent 也能合作（第九章）。

📖 登场：[尾声](#尾声协议让陌生人合作)　🧪 `capstone.py`

---

## 参考资料

- S. Crocker, *RFC 1: Host Software*, 1969-04-07；[RFC Editor](https://www.rfc-editor.org/rfc/rfc1.html)
- S. D. Crocker, *How the Internet Got Its Rules*, The New York Times, 2009-04-06
- V. Cerf, R. Kahn, *A Protocol for Packet Network Intercommunication*, IEEE Transactions on Communications 22(5), 1974
- T. Berners-Lee, [Information Management: A Proposal](https://info.cern.ch/Proposal.html), CERN, 1989-03
- CERN, [CERN puts the World Wide Web in the public domain](https://timeline.web.cern.ch/node/794), 1993-04-30
- Microsoft / Red Hat / Codenvy, Language Server Protocol, 2016-06；[规范](https://microsoft.github.io/language-server-protocol/)
- Anthropic, [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), 2024-11-25；[MCP 官方文档](https://modelcontextprotocol.io)
- Anthropic, [Donating the Model Context Protocol and establishing the Agentic AI Foundation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation), 2025-12-09
- Google, [Announcing the Agent2Agent Protocol (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/), 2025-04-09；[Google Cloud donates A2A to Linux Foundation](https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/), 2025-06-23
- A2A Project, [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)（本章按 1.0 版实现）；[GitHub](https://github.com/a2aproject/A2A)
- L. Chen, M. Zaharia, J. Zou, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, 2023；[arXiv:2305.05176](https://arxiv.org/abs/2305.05176)

---

⬅️ [第八章 多 Agent 协同](../08-multi-agent/)　|　[返回目录](../README.md)
