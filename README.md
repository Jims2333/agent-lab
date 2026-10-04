# 🤖 agent-lab

**边做边学：从 1958 年的感知机，一路学到今天的 Harness、多 Agent 协同，以及让 Claude、GPT、DeepSeek 一起干活。**

1958 年，《纽约时报》报道第一台会学习的机器时，用的标题是 *"New Navy Device Learns By Doing"*（边做边学）。
这个仓库沿用同一个思路：每一章都先讲**真实发生过的故事**，再**亲手跑代码**复现关键时刻，最后才解释**概念**。

## 学习路线

| 章节 | 内容 | 动手做什么 | 状态 |
|---|---|---|---|
| [01 机器学习从哪里来](./01-ml-origins/) | 1943–2026 的故事：感知机、AI 寒冬、反向传播、ImageNet、Transformer、Agent | 复现 1958 感知机、1969 XOR 困境、1986 反向传播 | ✅ |
| [02 线性模型与梯度下降](./02-linear-models/) | 1801–2014 的故事：找回谷神星、最小二乘之争、柯西下山法、高尔顿的"回归"、LMS、Adam | 用高尔顿的真实数据手写梯度下降，比较批量与随机梯度下降 | ✅ |
| [03 神经网络与自动求导](./03-neural-nets/) | 1970–2022 的故事：反向模式自动微分、万能逼近、梯度消失、ReLU、Dropout、深度学习框架 | 手写自动求导引擎并用它训练网络，观察 30 层网络里的梯度消失与爆炸 | ✅ |
| [04 从 CNN 到 Transformer](./04-cnn-to-transformer/) | 1959–2020 的故事：猫的视觉皮层、Neocognitron、ResNet、LSTM、注意力的诞生、Transformer、ViT | 手写卷积与注意力，比较 RNN 和注意力，组装一个完整的 Transformer | ✅ |
| [05 大语言模型与 API](./05-llm-and-api/) | 1913–2025 的故事：马尔可夫数字母、香农猜字母、词向量、BPE、规模定律、RLHF、DeepSeek-R1 | 训练 n-gram 模型和 BPE 分词器，玩转采样参数，用统一客户端调用 Claude / GPT / DeepSeek | ✅ |
| [06 Agent 循环](./06-agent-loop/) | 1966–2025 的故事：ELIZA、Shakey、SHRDLU、ReAct、函数调用、SWE-bench、Claude Code | 迷你 ELIZA、用 JSON Schema 定义并安全执行工具、从零写一个 ReAct agent | ✅ |
| [07 Harness 工程](./07-harness/) | 2024–2026 的故事：SWE-agent 的 ACI、Replit 删库事件、上下文工程、长时 agent 的交接、OpenAI 的 harness engineering | 给编程 agent 加上沙箱、权限闸门、预算、上下文整理、验收和日志；比较五种上下文管理策略 | ✅ |
| [08 多 Agent 协同](./08-multi-agent/) | 1973–2025 的故事：Actor 模型、Hearsay-II 的黑板、合同网、《心智社会》、2023 年的组队热潮、MAST 失败分类、Anthropic 与 Cognition 的争论 | 编排者-工作者并行调研；合同网投标分活；多 agent 辩论；写-审循环在子进程里跑测试 | ✅ |
| [09 跨模型协作](./09-cross-model/) | 1969–2026 的故事：浴室里写成的 RFC 1、TCP/IP、万维网、LSP、MCP、A2A | 在三家模型之间路由和故障转移；跨厂商交叉核对；用标准库实现迷你 A2A；毕业项目：三家模型分工做学习路线卡片 | ✅ |

**番外篇**

| 番外 | 内容 |
|---|---|
| [两个 AI 是怎么"聊天"的](./side-quests/ai-talks-to-ai/) | 让 Claude 和 DeepSeek / GPT 对话的原理：无状态 API、角色翻转、传话 harness、多 agent 分工、MCP / A2A。附可运行的 `relay.py` |
| [两个 App 之间的桥梁](./side-quests/app-bridge/) | 让 ChatGPT 和 Claude 两个桌面 App 互相传话：软件之间的通信方式、MCP 协议底层、本地与远程服务器、隧道。附可运行的 MCP 信箱服务器 |

## 快速开始

```bash
git clone https://github.com/Jims2333/agent-lab.git
cd agent-lab
pip install -r requirements.txt

# 第一章的三个实验
cd 01-ml-origins
python perceptron_1958.py
python xor_1969.py
python backprop_1986.py

# 第二章的四个实验
cd ../02-linear-models
python ceres_1801.py
python gradient_descent_1847.py
python galton_1886.py
python lms_1960.py

# 第三章的三个实验
cd ../03-neural-nets
python autograd.py
python mlp_moons.py
python vanishing_gradient.py

# 第四章的四个实验
cd ../04-cnn-to-transformer
python convolution.py
python attention.py
python rnn_vs_attention.py
python transformer_block.py

# 第五章的四个实验（默认用假模型，不需要 API key）
cd ../05-llm-and-api
python markov_1913.py
python bpe_tokenizer.py
python sampling.py
python chat_api.py

# 第六章的三个实验
cd ../06-agent-loop
python eliza_1966.py
python tools.py
python react_agent.py

# 第七章的两个实验
cd ../07-harness
python harness.py
python context_budget.py

# 第八章的四个实验
cd ../08-multi-agent
python orchestrator.py
python contract_net.py
python debate.py
python write_review.py

# 第九章的四个实验（加 --live 让配置了 key 的厂商用真实模型）
cd ../09-cross-model
python router.py
python cross_check.py
python a2a_mini.py
python capstone.py

# 番外 1：两个 AI 聊天（mock 模式，不需要 API key）
cd ../side-quests/ai-talks-to-ai
python relay.py --show-history

# 番外 2：看看 App 和 MCP 服务器之间说了什么（需要 pip install "mcp[cli]"）
cd ../app-bridge
python wire_demo.py
```

需要 Python 3.10+。前四章只依赖 numpy，第 5～9 章的假模型模式只用标准库。
从第 5 章开始可以接入真实的大模型（需要对应的 API key，只放在环境变量里），但所有实验默认都用假模型，不需要任何 key 也能跑通。

| 厂商 | 环境变量 | 默认型号 |
|---|---|---|
| Claude | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| GPT | `OPENAI_API_KEY` | `gpt-5.5` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |

> 关于真实模型模式：所有调用真实 API 的代码路径，作者都用模拟服务器测试过，但**没有用真实的 API key 跑过**。
> 如果你遇到问题，欢迎提 issue。

## 学完之后

九章走完，你手里已经有了一整套"玩具零件"：自动求导引擎、注意力、分词器、ReAct 循环、harness、多 agent 编排、适配层和迷你 A2A。接下来可以这样继续：

1. **换上真实模型重跑一遍**：配置一两家的 key，用 `--provider` 或 `--live` 运行第 5～9 章，对比真实模型和假模型剧本的差别。真实模型不会按剧本"犯错"，这本身就很有启发。
2. **建一个自己的小评测集**：第九章的路由表、交叉核对都依赖"哪家在哪类任务上够用"，这个问题只有你自己的测试数据能回答。
3. **把手写的零件换成正式工具**：自动求导换成 PyTorch，模型调用换成各家官方 SDK，MCP 用官方的 `mcp` Python SDK，A2A 用 `a2a-sdk`。因为你知道它们里面在做什么，换起来会很快。
4. **读原始资料**：每章末尾的参考资料都是一手来源，论文和原文往往比任何转述都精彩。
5. **做一个自己的项目**：比如把番外 2 的 MCP 信箱和第九章的 A2A 结合起来，或者给第七章的 harness 加一道"委托给外部 agent 之前先问人"的权限闸门。

## 写作约定

- **故事不编造**：人名、时间、地点、引语都有出处，每章末尾列出参考资料；只有单一来源的轶事会注明"据某某回忆"。
- **先故事，后概念**：概念统一放在每章最后解释，并标注它在哪个故事里登场。
- **代码从零写**：能用 numpy 手写的就不用框架，先看清原理，再用工具。
- **每个实验都能跑**：固定随机种子，输出可复现，脚本末尾附思考题。

## 目录结构

```
agent-lab/
├── README.md
├── requirements.txt
├── 01-ml-origins/              # 第一章：机器学习从哪里来
│   ├── README.md               #   故事 → 实验 → 概念 → 参考资料
│   ├── perceptron_1958.py
│   ├── xor_1969.py
│   └── backprop_1986.py
├── 02-linear-models/           # 第二章：线性模型与梯度下降
│   ├── README.md
│   ├── ceres_1801.py
│   ├── gradient_descent_1847.py
│   ├── galton_1886.py
│   ├── lms_1960.py
│   └── data/galton_1886.csv    #   高尔顿 1886 年的真实身高数据
├── 03-neural-nets/             # 第三章：神经网络与自动求导
│   ├── README.md
│   ├── autograd.py             #   手写的自动求导引擎（Value 类）
│   ├── mlp_moons.py
│   └── vanishing_gradient.py
├── 04-cnn-to-transformer/      # 第四章：从 CNN 到 Transformer
│   ├── README.md
│   ├── convolution.py
│   ├── attention.py
│   ├── rnn_vs_attention.py
│   └── transformer_block.py
├── 05-llm-and-api/            # 第五章：大语言模型与 API
│   ├── README.md
│   ├── markov_1913.py
│   ├── bpe_tokenizer.py
│   ├── sampling.py
│   ├── chat_api.py
│   └── data/corpus.txt         #   第 1～4 章的正文，当作训练语料
├── 06-agent-loop/              # 第六章：Agent 循环
│   ├── README.md
│   ├── eliza_1966.py
│   ├── tools.py                #   工具说明书 + 安全执行（后面几章也会用）
│   └── react_agent.py
├── 07-harness/                 # 第七章：Harness 工程
│   ├── README.md
│   ├── harness.py
│   └── context_budget.py
├── 08-multi-agent/             # 第八章：多 Agent 协同
│   ├── README.md
│   ├── orchestrator.py         #   工作者复用第六章的 ReAct agent
│   ├── contract_net.py
│   ├── debate.py
│   └── write_review.py
├── 09-cross-model/             # 第九章：跨模型协作
│   ├── README.md
│   ├── team.py                 #   三家厂商的名单：--live 时有 key 的用真实模型
│   ├── router.py
│   ├── cross_check.py
│   ├── a2a_mini.py             #   用标准库实现的 A2A 1.0 最小子集
│   └── capstone.py             #   毕业项目：复用第 6～9 章的零件
├── common/
│   └── llm.py                  # 第 5～9 章共用：统一调用 Claude / GPT / DeepSeek / 假模型
└── side-quests/
    ├── ai-talks-to-ai/         # 番外 1：两个 AI 是怎么聊天的
    │   ├── README.md
    │   └── relay.py
    └── app-bridge/             # 番外 2：ChatGPT 和 Claude 两个 App 之间的桥梁
        ├── README.md
        ├── bridge_server.py
        └── wire_demo.py
```
