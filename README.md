

# 🦀 CloseCrab — Secure Local AI Execution Engine

**CloseCrab** 是一个用 C++ 编写的本地 AI 执行引擎，专注于**安全、高性能、纯本地运行**。
无需云端，无需 Python，一键安装即可运行本地大模型。


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Windows](https://img.shields.io/badge/Platform-Windows-0078d7.svg)](https://www.microsoft.com/)
[![Release](https://img.shields.io/github/v/release/Blitzball996/CloseCrab)](https://github.com/Blitzball996/CloseCrab/releases)<a class="xsj_anchor xsj_anchor_range xsj_anchor_range_start" name="xsj_1774354231371"></a>
<a class="xsj_anchor xsj_anchor_range xsj_anchor_range_end" name="xsj_1774354231371"></a>

**CloseCrab** is a secure, high-performance local AI execution engine written in C++. It runs large language models on your own hardware, gives the AI the ability to execute skills (open apps, read/write files, search the web), and protects your system with a built-in security sandbox.

The gateway is just the interface — the core is your local AI that respects your privacy and never sends your data to the cloud.

Contact: kongshuangquan@gmail.com

---

## ✨ Features

| Feature                      | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| 🚀 **Local LLM Inference** | Run Qwen2.5-7B/14B/32B/Deepseek MoE Any models on your own GPU                     |
| 🎯 **GPU Acceleration**    | Full CUDA support with automatic layer allocation                                  |
| 📝 **Streaming Output**    | Real-time token-by-token response like ChatGPT                                     |
| 💾 **Conversation Memory** | SQLite-based session management with full history                                  |
| 🛠️ **Skill System**      | AI can execute real actions: open apps, read/write files, run commands, search web |
| 🔒 **Security Sandbox**    | 4 security modes (ASK/AUTO/TRUSTED/DISABLED) with audit logging                    |
| 🌐 **Multiple Access**     | CLI, WebSocket (ws://localhost:9001), HTTP API (REST)                              |
| 📚 **RAG Support**         | （FAISS）                                                                          |
| ⚙️ **Full Control**        | Configure GPU layers, batch size, threads, MoE expert allocation                   |
| ⚙️ **Lightweight**         | Single executable, no Python runtime, minimal dependencies                         |

---

## 🎯 Quick Start

### Download
```bash
# Download the latest release
wget https://github.com/yourusername/CloseCrab/releases/download/v1.0.0/CloseCrab_Setup.exe
```
## 安装


1.  下载安装包：

```
CloseCrab_Setup.exe
```

1.  双击安装
2.  安装过程中：
    👉 选择模型（自动下载）
    👉 自动写入配置
3.  安装完成后直接运行



## 🤖 模型说明

安装时可选：

| 模型 | 大小 | 推荐 |
| --- | --- | --- |
| Qwen2.5-7B | ~4.5GB | ⭐ 推荐 |
| Qwen2.5-14B | ~8.5GB | 高性能 |
| Qwen2.5-3B | ~2GB | 轻量 |
| Qwen2.5-1.5B | ~1.2GB | 极速 |

模型会自动下载到：

```
models/
```


## 配置自定义模型

编辑 `config/config.yaml`，修改 `llm.model_path` 指向你的模型文件：

 model_path: "models/your-model.gguf"
  
  注意如果运行报错Failed to Load Model，则需手动输入config.yaml中modelpath并保存
  

## ▶️ 运行

双击：

```
closecrab.exe
```

或桌面图标


## 💬 常用命令

*   `/help` - 显示帮助

*   `/skills` - 查看可用技能

*   `/skillmode auto` - AI 自动决定是否使用技能

*   `/skillmode chat` - 纯聊天模式

*   `/skillmode ask` - 使用技能前询问

*   `/clear` - 清空对话历史

*   `/new` - 开始新会话

*   `/quit` - 退出


## 🛠️ 技能系统

*   open_app
*   read_file
*   write_file
*   execute_command
*   search_web
*   screenshot
*   systeminfo
*   clipboard
*   weather
*   ReadBinaryFile


# ⚙️ 配置

配置文件：

```
config/config.yaml
```

核心字段：

```
llm:
  model_path: "models/xxx.gguf"
```



# 🧱 构建（开发者）

## 环境

*   Windows 10/11
*   Visual Studio 2022
*   CMake 3.20+
*   CUDA 12.x（可选）

* * *

## 依赖

*   llama.cpp（本地编译）
*   FAISS（vcpkg）
*   libcurl（vcpkg）
*   OpenSSL
*   ZLIB



## 依赖

*   llama.cpp（本地编译）
*   FAISS（vcpkg）
*   libcurl（vcpkg）
*   OpenSSL
*   ZLIB


### 下载 llama.cpp 并编译 CUDA 版本

powershell
```powershell
cd external
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=ALL
cmake --build . --config Release
```


## 安装 vcpkg 依赖

```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install curl:x64-windows
.\vcpkg install faiss:x64-windows
```


## 构建

```
cmake -B build ^
 -DCMAKE_TOOLCHAIN_FILE=G:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
 -DVCPKG_TARGET_TRIPLET=x64-windows ^
 -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```


## 🚨 重要说明（必须看）

## ❗ 1\. 只使用 Release

CloseCrab 为高性能推理引擎：

*   ❌ 不建议使用 Debug
*   ❌ Debug 可能缺少 libcurl-d.dll
*   ✅ 使用 Release 或 RelWithDebInfo



## ❗ 2\. 运行时报错缺 DLL

如果运行报错：

```
缺少 libcurl.dll / faiss.dll
```

👉 原因：缺少运行时依赖

### 解决：

确保这些 DLL 在 exe 同目录：

*   libcurl.dll
*   faiss.dll
*   openblas.dll
*   zlib1.dll

👉 推荐：通过 CMake 自动拷贝



## ❗ 3\. GPU 不生效

检查：

*   CUDA 是否安装
*   llama.cpp 是否 CUDA 编译
*   是否设置：

```
n_gpu_layers = -1
```


# 📦 安装包说明

安装器支持：

*   自动拷贝所有 DLL
*   自动下载模型（可选）
*   自动生成 config.yaml
*   创建桌面快捷方式



# 📁 目录结构

```
CloseCrab/
├── src/
├── config/
├── models/
├── data/
├── external/
├── build/
└── closecrab.exe
```

# 🧠 项目定位

CloseCrab 不是 Demo，而是：

👉 **本地 AI Agent 执行引擎**

目标：

*   替代云 AI
*   本地隐私计算
*   可扩展技能系统
*   高性能推理
