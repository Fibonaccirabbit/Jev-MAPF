<div align="center">

<img src="docs/banner.svg" alt="JEV//MAPF" width="100%">

### 让大模型指挥一群机器人，在迷宫里互不挡道。

**🗺️ 149 张地图开箱即跑 · 🧠 云端 API / 本地小模型 · 🎮 每一步选择都看得见**

无需 GPU，无需训练。装好就能打开浏览器，看每个机器人怎么观察、怎么选、怎么走。

[![CI](https://github.com/Fibonaccirabbit/Jev-MAPF/actions/workflows/test.yml/badge.svg)](https://github.com/Fibonaccirabbit/Jev-MAPF/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-ffb400?style=flat)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![POGEMA](https://img.shields.io/badge/Env-POGEMA-ff2bd6)](https://github.com/Cognitive-AI-Systems/pogema)

[English](README.md) · **简体中文**

[🚀 快速上手](#-快速上手) · [🧠 接入模型](#-给机器人接上模型) · [📊 实测结果](#-实测结果) · [🔧 继续折腾](#-想继续折腾)

<img src="docs/demo.gif" alt="DeepSeek + 协作规划在 puzzle-00 上以最优 20 步完成" width="100%">

<sub>DeepSeek × 协作规划 · 官方 <code>puzzle-00</code> · 4 个机器人 · 20 步完成 · 右栏是每个机器人每步的耗时、token 与候选动作</sub>

如果它帮你省下了搭环境的时间，欢迎点一颗 **⭐ Star**。

</div>

## ✨ 打开工作台，你能做什么

- 🗺️ **先跑起来，再接模型**：内置 A\* 与随机基线，没有 Key 也能跑完整流程、播放预置回放。
- 🧠 **模型入口在界面里**：DeepSeek、TypeSafe Jev、本地 Qwen RLCD、任意 OpenAI 兼容接口，填完即可测试调用。
- 👀 **决策过程看得见**：每个机器人这一步的耗时、token、五个候选动作、实际选择；接口返回原生概率时显示概率条。
- 🎚️ **三个开关调难度**：自身地图距离、短程无线电、协作规划，逐层打开看效果变化。
- ⏯️ **随手控制实验**：运行、单步、暂停、停止、时间轴回放、倍速播放、JSON 导出。
- 📦 **随包带回放**：9 段运行记录，点开就能看成功与死锁的差别。

## 🎯 149 张地图，四个机器人

| 地图 | 你会看到什么 |
| --- | --- |
| 🧩 puzzle（16 张） | 5×5 窄通道，必须有机器人退进侧袋让路，否则互相卡死 |
| 🌀 maze（128 张） | 21×21 迷宫，考验绕路与长距离导航 |
| 🏭 warehouse（1 张） | 33×46 货架走廊 |
| 🔰 调试场景（4 个） | 交叉、会车、四房间、随机障碍，适合先熟悉界面 |

机器人到达目标后仍然占格，需要时可以移开让路 —— 这是经典 MAPF 设定，也是互相挡路的来源。

## 🚀 快速上手

**准备好 Python 3.11+ 和 Git。** 不需要 GPU，也不需要前端构建。

### 1️⃣ 下载项目

```bash
git clone https://github.com/Fibonaccirabbit/Jev-MAPF.git
cd Jev-MAPF
```

### 2️⃣ 安装并启动

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/jev-mapf serve --port 8091
```

### 3️⃣ 打开页面，跑第一个实验

访问 **[http://127.0.0.1:8091](http://127.0.0.1:8091)** → 底部点开任意一段回放，或选 **A\*** 基线 → 点 **运行**。🎉

拖动时间轴回看任意一步，右栏会跟着显示那一步每个机器人的候选与选择。

> 💡 A\* 与随机基线不调用模型。接入模型后，每个机器人每步产生一次调用。

## 🧠 给机器人接上模型

| 方式 | 适合谁 | 需要准备 |
| --- | --- | --- |
| 🎮 A\* · 随机基线 | 想先看界面和地图 | 装好项目即可 |
| 🔵 DeepSeek | 手上有 DeepSeek Key | API Key |
| ⚡ TypeSafe Jev | 已获邀请的 TypeSafe 用户 | 官方 API Key |
| 🍎 本地 Qwen2.5-1.5B RLCD | Apple Silicon 上跑 4-bit 约束解码 | 启动本地 Qwen RLCD 服务 |
| 🌐 OpenAI 兼容 API | 已有云端平台或本地服务 | 请求地址、模型 ID、API Key |

点顶栏 **⚙** → 填地址、模型 ID、API Key → 保存 → **测试**，通过后显示接口返回的模型名与延迟。

| 模型 | `--provider` | 环境变量 |
| --- | --- | --- |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` · `DEEPSEEK_MODEL` |
| TypeSafe Jev | `jev` | `TYPESAFE_API_KEY` · `JEV_MAX_ATTEMPTS` |
| Qwen RLCD | `qwen_rlcd` | `QWEN_RLCD_URL`（默认 `http://127.0.0.1:8000/api/run-rlcd`） |
| OpenAI 兼容 | `chat` | `MAPF_API_URL` · `MAPF_API_MODEL` · `MAPF_API_KEY` |

DeepSeek 与 Jev 固定使用各自官方地址。DeepSeek 默认开启 low 思考、8192 token 预算；puzzle 上改成 `--no-thinking --max-tokens 1024` 更快更省。Jev 遇到连接失败会按 `JEV_MAX_ATTEMPTS` 重试。

## 🔄 它是怎么工作的

```text
  each robot, every tick, in parallel and isolated
  +--------------------------------------------------------------------+
  |  7x7 sensors -> private memory -> local facts -> <=5 moves -> MODEL |
  +--------------------------------------------------------------------+
        ^                                ^
        | radio . range r                | coop planner . range 2r . opt-in
        | priority + wanted cell         | pooled maps + goals -> joint A*
        +----------------+---------------+
                         |
                         v
      POGEMA synchronous step   (soft collisions, no swaps)
```

每个机器人只看自己半径 3 的 7×7 视野，加上自己走过的地图记忆和最近 8 步反馈，从最多 5 个一步动作里选一个。所有选择凑齐后一次性交给 POGEMA 执行。三个开关决定它还能知道什么：

| 开关 | 默认 | 机器人多知道什么 | 代码 |
| --- | :-: | --- | --- |
| **自身地图距离** | on | 各候选格在自己已观测地图上离目标还有几步；未知格按可通行算 | `perception.py` |
| **短程无线电** | on | 视野内邻居的优先级和它下一步想去哪，据此判断该让谁 | `comms.py` |
| **协作规划** | off | 无线电范围内的机器人凑一组，共享地图与目标做联合搜索，给出本步建议 | `planner.py` |

<details>
<summary>💡 为什么需要协作规划？</summary>

<br>

走廊里迎面相遇的两个机器人执行同一套规则，会同时让开、又同时回来，一直耗到步数上限。无线电的优先级能打破一部分对称，但 puzzle 要的是“先退进侧袋、等对方过去”这种多步前瞻，反应式规则的天花板大致就是 PIBT。协作规划让无线电范围内的小组跑一次联合搜索，第一步作为建议交给模型。组内共享了目标，所以这类结果单列为「去中心化 + 协作规划」。

</details>

## 📊 实测结果

16 张官方 puzzle，4 个机器人，seed 42，32 步上限。联合状态穷举显示 15 张可解（`puzzle-06` 无解）。

```text
  独立 A*              ████░░░░░░░░░░░░   4/16   ISR 0.69
  DeepSeek · 单打独斗   ███░░░░░░░░░░░░░   3/16   ISR 0.66
  DeepSeek · 协作规划   ███████████████░  15/16   ISR 0.97   全部最优步数
  Jev      · 协作规划   ███████████████░  15/16   ISR 0.94   全部最优步数
  Qwen 1.5B · 协作规划  ░░░░░░░░░░░░░░░░   0/16   ISR 0.12
```

| | DeepSeek × 协作 | Jev × 协作 |
| --- | --: | --: |
| 解出（均为最优步数） | **15 / 16** | **15 / 16** |
| 采纳计划建议 | 516 / 516 | 516 / 516 |
| 单次调用中位延迟 | 728 ms | 590 ms |
| 16 张图总 tokens | 1.52 M | 1.51 M |

puzzle 考验的是多步协同让行：独立 A\*、PIBT 和逐步 LLM 决策都停在 3–5 张。Qwen 1.5B 能跑通协议，但基本不跟随距离信号，从第一步起两格振荡。

完整过程、失败机制、中间版本与网络故障记录见 **[实验记录](EXPERIMENTS.md)**。

## 🖥️ 命令行

```bash
# 基线
jev-mapf run --case puzzle-03 --provider astar --num-agents 4

# DeepSeek × 协作规划
DEEPSEEK_API_KEY=sk-... jev-mapf run --case puzzle-00 --provider deepseek \
  --no-thinking --max-tokens 1024 --num-agents 4 --max-steps 32 --coop-planner

# Jev 跑官方迷宫，从隐藏输入读 Key
jev-mapf run --case validation-mazes-seed-000 --provider jev --num-agents 4 --max-steps 96 --key-stdin

# 把一次运行打包成界面里的预置回放
jev-mapf preset outputs/<run-id> --name my-run --title "DeepSeek × 协作 · puzzle-00"
```

参数：`--seed` · `--max-steps 1–256` · `--num-agents 1–16` · `--obs-radius 1–5` · `--coop-planner` · `--no-thinking` · `--max-tokens` · `--key-stdin`。

每次运行写入 `outputs/<run-id>/`：`result.json`（配置、地图来源、逐帧状态、每个机器人的输入与候选、请求与返回、延迟、选择、执行结果）、`animation.svg`、`replay.html`。CSR 是这一局有没有全部到达，ISR 是结束时到达目标的比例，SoC 与 makespan 取自 POGEMA。

## 🔧 想继续折腾

换地图、加机器人、调视野半径，界面里都能改。要接新模型或改提示词，从这几个文件入手：

| 文件 | 负责什么 |
| --- | --- |
| `decentralized.py` | 每个机器人的记忆、候选、提示词、并发请求 |
| `perception.py` · `comms.py` · `planner.py` | 三层局部信息 |
| `policies.py` | 各家模型协议与基线 |
| `runtime.py` | POGEMA 执行循环与记录 |
| `server.py` · `web/` | 工作台 API 与霓虹界面（原生 JS，无构建） |

```bash
.venv/bin/python -m pytest -q
npm i --no-save playwright && npx playwright install chromium
node tests/ui_smoke.cjs      # 浏览器冒烟：预置回放 + A*
```

## 🗺️ 接下来

- [x] 149 张地图、局部观测、独立决策、同步执行
- [x] DeepSeek / Jev / Qwen RLCD / OpenAI 兼容接口
- [x] 自身地图距离、短程无线电、协作规划三层开关
- [x] 霓虹工作台：逐步决策面板、时间轴回放、预置回放
- [x] 16 张 puzzle 的三模型实测
- [ ] 迷宫与 warehouse 地图的完整实测
- [ ] 多 seed 成功率统计
- [ ] 更大规模（8–16 个机器人）的协作规划

## 🤝 一起维护

欢迎 ⭐ Star、🍴 Fork，也欢迎提 Issue 和 PR：接入新模型、复现失败案例、补充地图、改进界面都很有用。

[🐛 报告问题](https://github.com/Fibonaccirabbit/Jev-MAPF/issues) · [🛠️ 提交 PR](https://github.com/Fibonaccirabbit/Jev-MAPF/pulls)

提交实验结果时带上模型、地图、seed 和配置，成功和失败都值得记录。

## 🙏 致谢与许可

仿真环境与 A\* 基线来自 [POGEMA](https://github.com/Cognitive-AI-Systems/pogema)（MIT），随包地图的许可与来源见 [`src/jev_mapf/maps/`](src/jev_mapf/maps/)。本项目原创代码采用 **[MIT](LICENSE)**。
