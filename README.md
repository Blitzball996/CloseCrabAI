

# 🦀 CloseCrab — Secure Local AI Execution Engine—安全本地 AI 智能体引擎

**CloseCrab** 是一个用 C++ 编写的本地 AI 执行引擎，专注于**安全、高性能、纯本地运行**。它让你在本地运行大语言模型，并赋予 AI 执行实际操作的技能，同时通过安全沙箱保护你的系统。

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
| 🚀 **Local LLM Inference** | Run Qwen2.5-7B/14B/32B/Any models on your own GPU                                  |
| 🎯 **GPU Acceleration**    | Full CUDA support with automatic layer allocation                                  |
| 📝 **Streaming Output**    | Real-time token-by-token response like ChatGPT                                     |
| 💾 **Conversation Memory** | SQLite-based session management with full history                                  |
| 🛠️ **Skill System**      | AI can execute real actions: open apps, read/write files, run commands, search web |
| 🔒 **Security Sandbox**    | 4 security modes (ASK/AUTO/TRUSTED/DISABLED) with audit logging                    |
| 🌐 **Multiple Access**     | CLI, WebSocket (ws://localhost:9001), HTTP API (REST)                              |
| 📚 **RAG Support**         | Load documents, vector search, retrieve relevant context                           |
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

1\. 下载 CloseCrab_Setup.exe
2\. 双击运行安装程序
3\. 按照提示完成安装

## 下载模型

安装后，运行 `download_model.bat` 选择要下载的模型，或手动下载 GGUF 模型放到 `models` 文件夹。

推荐模型：
- Qwen2.5-7B-Instruct-Q4_K_M.gguf (约 4.5GB)
- Qwen2.5-14B-Instruct-Q4_K_M.gguf (约 8.5GB)

下载地址：https://huggingface.co/Qwen

## 配置模型

编辑 `config/config.yaml`，修改 `llm.model_path` 指向你的模型文件：

 model_path: "models/your-model.gguf"
  
  
## 运行

双击桌面图标或运行 `closecrab.exe`

## 命令

*   `/help` - 显示帮助

*   `/skills` - 查看可用技能

*   `/skillmode auto` - AI 自动决定是否使用技能

*   `/skillmode chat` - 纯聊天模式

*   `/skillmode ask` - 使用技能前询问

*   `/clear` - 清空对话历史

*   `/new` - 开始新会话

*   `/quit` - 退出

## 技能

*   `open_app` - 打开应用程序

*   `read_file` - 读取文件

*   `write_file` - 写入文件

*   `execute_command` - 执行系统命令

*   `search_web` - 网络搜索

*   `screenshot` - 屏幕截取

*   `systeminfo` - 获取系统信息

*   `clipboard`   - 操作剪贴板

*   `weather`       - 获取天气信息

*   `ReadBinaryFile` - 读取二进制文件
# 构建 CloseCrab

## 系统要求

*   Windows 10/11

*   NVIDIA GPU（可选，有 GPU 更快）

*   8GB+ 内存

*   4GB+ 磁盘空间（模型文件）
- Windows 10/11
- Visual Studio 2022 (社区版或更高)
- CMake 3.20+
- Git
- NVIDIA GPU (可选，用于 GPU 加速)
- CUDA Toolkit 12.6 (如需 GPU 支持)

## 依赖库

CloseCrab 使用 CMake FetchContent 自动下载大部分依赖，但部分库需要手动准备：

- **llama.cpp**：用于本地 LLM 推理。建议手动克隆到 `external/llama.cpp`，并编译 CUDA 版本。
- **OpenSSL**：需要手动安装 (用于 WebSocket)。下载地址：https://slproweb.com/products/Win32OpenSSL.html (选择 Win64 OpenSSL v3.6.1)
- **ZLIB**：用于压缩，通过 CMake 自动下载，或手动编译。
- **libcurl**：用于网络搜索，通过 vcpkg 安装。

### 下载 llama.cpp 并编译 CUDA 版本

powershell
```powershell
cd external
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build . --config Release
```

### 安装 vcpkg 和 libcurl

```powershell
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
.\vcpkg install curl:x64-windows
```
## 构建步骤

### 1\. 克隆仓库

powershell

```powershell
git clone https://github.com/yourname/CloseCrab.git
cd CloseCrab
```

### 2\. 准备依赖

*   安装 OpenSSL（默认路径 `C:\Program Files\OpenSSL-Win64`）

*   安装 vcpkg 并安装 curl（如上）

*   编译 llama.cpp（如上）

### 3\. 配置 CMake

打开 Visual Studio 2022，选择“打开本地文件夹”，选择 CloseCrab 目录。CMake 会自动配置项目。

或者使用命令行：

```cmd
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=G:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

> 
> 
> 注意：`-DCMAKE_TOOLCHAIN_FILE` 指向 vcpkg 的工具链文件，以便找到 libcurl。
> 
> 

### 4\. 编译

在 Visual Studio 中，选择生成菜单 → 全部生成。或命令行：

```cmd
cmake --build build --config Release
```

编译完成后，可执行文件位于 `build/Release/closecrab.exe`。

### 5\. 运行

将 `config` 文件夹复制到 exe 同级目录，下载模型文件放入 `models` 文件夹，编辑 `config/config.yaml` 指定模型路径，然后运行 `closecrab.exe`。

## 常见问题

### Q: 编译时找不到 curl/curl.h

A: 确保 vcpkg 已正确安装 curl，并在 CMake 配置时使用 vcpkg 工具链。如果手动指定，可在 CMakeLists.txt 中设置 `CURL_INCLUDE_DIR` 和 `CURL_LIBRARY`。

### Q: 运行时缺少 DLL

A: 将 `external\llama.cpp\build\bin\Release\llama.dll` 复制到 exe 目录。如果是 CUDA 版本，还需要 `cudart64_12.dll` 等。

### Q: GPU 加速未启用

A: 确保使用 CUDA 编译的 llama.dll，并在 `LLMEngine.cpp` 中设置 `model_params.n_gpu_layers = -1`。

### Q: 如何切换 Debug/Release

在 Visual Studio 顶部工具栏选择配置，或命令行指定 `--config Debug`。

## 目录结构

text

<pre>CloseCrab/
├── CMakeLists.txt
├── config/
│   └── config.yaml
├── src/
│   ├── main_interactive.cpp
│   ├── config/
│   ├── core/
│   ├── llm/
│   ├── memory/
│   ├── network/
│   ├── security/
│   ├── skills/
│   └── rag/
├── external/
│   ├── llama.cpp/
│   ├── ixwebsocket/
│   └── ...
├── models/            (用户放置模型文件)
├── data/              (运行时数据库)
└── build/             (编译输出)</pre>
