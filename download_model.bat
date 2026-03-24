@echo off
chcp 65001 > nul
echo ========================================
echo CloseCrab Model Downloader
echo ========================================
echo.
echo 请选择要下载的模型:
echo.
echo [1] Qwen2.5-7B (推荐, 4.5GB) - 适合 8GB 显存
echo     日常使用，速度快，中文能力强
echo.
echo [2] Qwen2.5-14B (更强, 8.5GB) - 适合 12GB 显存
echo     推理能力更强，适合复杂任务
echo.
echo [3] Qwen2.5-3B (轻量, 2GB) - 适合 6GB 显存
echo     速度快，显存占用低
echo.
echo [4] Qwen2.5-1.5B (极速, 1.2GB) - 适合 4GB 显存
echo     响应极快，适合简单对话
echo.
set /p choice="请输入数字 (1-4): "

if "%choice%"=="1" (
    set MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
    set MODEL_NAME=qwen2.5-7b-instruct-q4_k_m.gguf
    set MODEL_SIZE=4.5GB
    set VRAM=8GB
)
if "%choice%"=="2" (
    set MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf
    set MODEL_NAME=qwen2.5-14b-instruct-q4_k_m.gguf
    set MODEL_SIZE=8.5GB
    set VRAM=12GB
)
if "%choice%"=="3" (
    set MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
    set MODEL_NAME=qwen2.5-3b-instruct-q4_k_m.gguf
    set MODEL_SIZE=2GB
    set VRAM=6GB
)
if "%choice%"=="4" (
    set MODEL_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
    set MODEL_NAME=qwen2.5-1.5b-instruct-q4_k_m.gguf
    set MODEL_SIZE=1.2GB
    set VRAM=4GB
)

echo.
echo 正在下载 %MODEL_NAME% (%MODEL_SIZE%)
echo 需要 %VRAM% 显存
echo 请耐心等待...
echo.

mkdir models 2>nul
curl -L -o "models\%MODEL_NAME%" "%MODEL_URL%"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo 下载完成！
    echo 模型文件: models\%MODEL_NAME%
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 下载失败！请检查网络后重试。
    echo ========================================
)

pause