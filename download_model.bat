@echo off
chcp 65001 > nul
echo ========================================
echo CloseCrab Model Downloader
echo ========================================
echo.

:: ============================================
:: 第一部分：选择 LLM 模型
:: ============================================
echo [第1步] 选择 LLM 大语言模型:
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
    set LLM_URL=https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
    set LLM_NAME=qwen2.5-7b-instruct-q4_k_m.gguf
    set LLM_SIZE=4.5GB
)
if "%choice%"=="2" (
    set LLM_URL=https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf
    set LLM_NAME=qwen2.5-14b-instruct-q4_k_m.gguf
    set LLM_SIZE=8.5GB
)
if "%choice%"=="3" (
    set LLM_URL=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
    set LLM_NAME=qwen2.5-3b-instruct-q4_k_m.gguf
    set LLM_SIZE=2GB
)
if "%choice%"=="4" (
    set LLM_URL=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
    set LLM_NAME=qwen2.5-1.5b-instruct-q4_k_m.gguf
    set LLM_SIZE=1.2GB
)

:: ============================================
:: 第二部分：RAG 模型（Embedding + Reranker）
:: ============================================
echo.
echo ========================================
echo [第2步] RAG 检索增强模型
echo ========================================
echo.
echo RAG 模型用于知识库检索，提升回答质量。
echo.
echo   Embedding: BAAI/bge-small-zh-v1.5 (134MB, 支持中文)
echo   Reranker:  BAAI/bge-reranker-base  (1.1GB)
echo.
set /p rag_choice="是否下载 RAG 模型? (Y/n): "
if /i "%rag_choice%"=="n" (
    set DOWNLOAD_RAG=0
) else (
    set DOWNLOAD_RAG=1
)

:: ============================================
:: 创建目录
:: ============================================
mkdir models 2>nul
mkdir models\bge-small-zh 2>nul
mkdir models\bge-reranker-base 2>nul

:: ============================================
:: 下载 LLM
:: ============================================
echo.
echo ========================================
echo 正在下载 LLM: %LLM_NAME% (%LLM_SIZE%)
echo ========================================
echo.

curl -L --progress-bar -o "models\%LLM_NAME%" "%LLM_URL%"

if %errorlevel% equ 0 (
    echo [OK] LLM 下载完成: models\%LLM_NAME%
) else (
    echo [FAIL] LLM 下载失败！请检查网络后重试。
)

:: ============================================
:: 下载 RAG 模型
:: ============================================
if "%DOWNLOAD_RAG%"=="1" (
    echo.
    echo ========================================
    echo 正在下载 Embedding 模型: bge-small-zh-v1.5
    echo ========================================
    echo.

    echo [1/3] 下载 model.onnx (约 90MB)...
    curl -L --progress-bar -o "models\bge-small-zh\model.onnx" ^
        "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/onnx/model.onnx"

    if %errorlevel% equ 0 (
        echo [OK] Embedding model.onnx 下载完成
    ) else (
        echo [FAIL] Embedding model.onnx 下载失败
    )

    echo [2/3] 下载 vocab.txt...
    curl -L --progress-bar -o "models\bge-small-zh\vocab.txt" ^
        "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/vocab.txt"

    if %errorlevel% equ 0 (
        echo [OK] Embedding vocab.txt 下载完成
    ) else (
        echo [FAIL] Embedding vocab.txt 下载失败
    )

    echo.
    echo ========================================
    echo 正在下载 Reranker 模型: bge-reranker-base
    echo ========================================
    echo.

    echo [3/3] 下载 model.onnx (约 1.1GB)...
    curl -L --progress-bar -o "models\bge-reranker-base\model.onnx" ^
        "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/onnx/model.onnx"

    if %errorlevel% equ 0 (
        echo [OK] Reranker model.onnx 下载完成
    ) else (
        echo [FAIL] Reranker model.onnx 下载失败
    )

    echo 下载 vocab.txt...
    curl -L --progress-bar -o "models\bge-reranker-base\vocab.txt" ^
        "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/vocab.txt"

    if %errorlevel% equ 0 (
        echo [OK] Reranker vocab.txt 下载完成
    ) else (
        echo [FAIL] Reranker vocab.txt 下载失败
    )
)

:: ============================================
:: 完成
:: ============================================
echo.
echo ========================================
echo 下载汇总:
echo ========================================
echo   LLM:       models\%LLM_NAME%
if "%DOWNLOAD_RAG%"=="1" (
    echo   Embedding: models\bge-small-zh\model.onnx
    echo              models\bge-small-zh\vocab.txt
    echo   Reranker:  models\bge-reranker-base\model.onnx
    echo              models\bge-reranker-base\vocab.txt
) else (
    echo   RAG 模型: 跳过（可稍后运行此脚本重新下载）
)
echo ========================================
echo.

pause