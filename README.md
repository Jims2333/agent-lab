# 🤖 agent-lab

**边做边学：从 1958 年的感知机，一路学到今天的 Harness 与多 Agent 协同。**

1958 年，《纽约时报》报道第一台会学习的机器时，用的标题是 *"New Navy Device Learns By Doing"*（边做边学）。
这个仓库沿用同一个思路：每一章都先讲**真实发生过的故事**，再**亲手跑代码**复现关键时刻，最后才解释**概念**。

## 学习路线

| 章节 | 内容 | 动手做什么 | 状态 |
|---|---|---|---|
| [01 机器学习从哪里来](./01-ml-origins/) | 1943–2026 的故事：感知机、AI 寒冬、反向传播、ImageNet、Transformer、Agent | 复现 1958 感知机、1969 XOR 困境、1986 反向传播 | ✅ |
| [02 线性模型与梯度下降](./02-linear-models/) | 1801–2014 的故事：找回谷神星、最小二乘之争、柯西下山法、高尔顿的"回归"、LMS、Adam | 用高尔顿的真实数据手写梯度下降，比较批量与随机梯度下降 | ✅ |
| [03 神经网络与自动求导](./03-neural-nets/) | 1970–2022 的故事：反向模式自动微分、万能逼近、梯度消失、ReLU、Dropout、深度学习框架 | 手写自动求导引擎并用它训练网络，观察 30 层网络里的梯度消失与爆炸 | ✅ |
| [04 从 CNN 到 Transformer](./04-cnn-to-transformer/) | 1959–2020 的故事：猫的视觉皮层、Neocognitron、ResNet、LSTM、注意力的诞生、Transformer、ViT | 手写卷积与注意力，比较 RNN 和注意力，组装一个完整的 Transformer | ✅ |
| 05 大语言模型与 API | token、上下文窗口、调用 Claude / GPT / DeepSeek | 用 API 做第一个小应用 | 🚧 |
| 06 Agent 循环 | ReAct、工具调用：从"会说"到"会做" | 从零写一个能调用工具的 agent | 🚧 |
| 07 Harness 工程 | 上下文管理、权限、反馈回路、测试 | 给 agent 加上沙箱、检查器和自动重试 | 🚧 |
| 08 多 Agent 协同 | 编排者-工作者、路由、写-审、辩论 | 搭一个主 agent + 多个子 agent 的调研系统 | 🚧 |
| 09 跨模型协作 | Claude × GPT × DeepSeek，MCP 与 A2A | 让不同厂商的模型分工完成一个任务 | 🚧 |

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

# 番外 1：两个 AI 聊天（mock 模式，不需要 API key）
cd ../side-quests/ai-talks-to-ai
python relay.py --show-history

# 番外 2：看看 App 和 MCP 服务器之间说了什么（需要 pip install "mcp[cli]"）
cd ../app-bridge
python wire_demo.py
```

需要 Python 3.10+。前几章只依赖 numpy，从第 5 章开始才需要各家的 API key。

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
└── side-quests/
    ├── ai-talks-to-ai/         # 番外 1：两个 AI 是怎么聊天的
    │   ├── README.md
    │   └── relay.py
    └── app-bridge/             # 番外 2：ChatGPT 和 Claude 两个 App 之间的桥梁
        ├── README.md
        ├── bridge_server.py
        └── wire_demo.py
```
