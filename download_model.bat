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
echo [1] Qwen2.5-7B  (推荐, 4.5GB, 8GB显存)
echo [2] Qwen2.5-14B (更强, 8.5GB, 12GB显存)
echo [3] Qwen2.5-3B  (轻量, 2GB,  6GB显存)
echo [4] Qwen2.5-1.5B(极速, 1.2GB, 4GB显存)
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
:: 第二部分：RAG 模型
:: ============================================
echo.
echo ========================================
echo [第2步] RAG 检索增强模型
echo ========================================
echo.
echo 来自 onnx-community 预转换仓库:
echo   Embedding: bge-small-zh-v1.5-ONNX  (~96MB)
echo   Reranker:  bge-reranker-base-ONNX  (~1.1GB)
echo   每个模型下载: model.onnx + model.onnx_data + tokenizer.json
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
mkdir models\bge-small-zh\onnx 2>nul
mkdir models\bge-reranker-base 2>nul
mkdir models\bge-reranker-base\onnx 2>nul

:: ============================================
:: 下载 LLM
:: ============================================
echo.
echo ========================================
echo 下载 LLM: %LLM_NAME% (%LLM_SIZE%)
echo ========================================
curl -L --retry 3 --retry-delay 5 --progress-bar -o "models\%LLM_NAME%" "%LLM_URL%"
if %errorlevel% equ 0 ( echo [OK] LLM 完成 ) else ( echo [FAIL] LLM 失败 )

:: ============================================
:: 下载 RAG 模型
:: ============================================
if "%DOWNLOAD_RAG%"=="1" (
    echo.
    echo ========================================
    echo 下载 Embedding: bge-small-zh-v1.5-ONNX
    echo ========================================

    echo [1/6] model.onnx...
    curl -L --retry 3 --retry-delay 5 --progress-bar -o "models\bge-small-zh\onnx\model_quantized.onnx" ^
        "https://huggingface.co/onnx-community/bge-small-zh-v1.5-ONNX/resolve/main/onnx/model_quantized.onnx
"
    if %errorlevel% equ 0 ( echo       [OK] ) else ( echo       [FAIL] )

    echo [2/6] model.onnx_data...
    curl -L --retry 3 --retry-delay 5 --progress-bar -o "models\bge-small-zh\onnx\model_quantized.onnx_data" ^
        "https://huggingface.co/onnx-community/bge-small-zh-v1.5-ONNX/resolve/main/onnx/model_quantized.onnx_data"
    if %errorlevel% equ 0 ( echo       [OK] ) else ( echo       [INFO] 可能不存在 )

    echo [3/6] tokenizer.json...
    curl -L --retry 3 --retry-delay 5 --progress-bar -o "models\bge-small-zh\tokenizer.json" ^
        "https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/tokenizer.json"
    if %errorlevel% equ 0 ( echo       [OK] ) else ( echo       [FAIL] )

    echo.
    echo ========================================
    echo 下载 Reranker: bge-reranker-base-ONNX
    echo ========================================

    echo [4/6] model.onnx (约1.1GB, 请耐心等待)...
    curl -L --retry 3 --retry-delay 5 --progress-bar -o "models\bge-reranker-base\onnx\model_uint8.onnx
" ^
        "https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/onnx/model_uint8.onnx
"
    if %errorlevel% equ 0 ( echo       [OK] ) else ( echo       [FAIL] )

    echo [5/6] model.onnx_data...
    curl -L --retry 3 --retry-delay 5 --progress-bar -o "models\bge-reranker-base\onnx\model_uint8.onnx
_data" ^
        "https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/onnx/model_uint8.onnx
_data"
    if %errorlevel% equ 0 ( echo       [OK] ) else ( echo       [INFO] 可能不存在 )

    echo [6/6] tokenizer.json...
    curl -L --retry 3 --retry-delay 5 --progress-bar -o "models\bge-reranker-base\tokenizer.json" ^
        "https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/tokenizer.json"
    if %errorlevel% equ 0 ( echo       [OK] ) else ( echo       [FAIL] )
)

:: ============================================
:: 完成
:: ============================================
echo.
echo ========================================
echo 下载汇总:
echo ========================================
echo   LLM: models\%LLM_NAME%
if "%DOWNLOAD_RAG%"=="1" (
    echo   Embedding:
    echo     models\bge-small-zh\onnx\model_quantized.onnx
    echo     models\bge-small-zh\onnx\model_quantized.onnx_data
    echo     models\bge-small-zh\tokenizer.json
    echo   Reranker:
    echo     models\bge-reranker-base\onnx\model_uint8.onnx

    echo     models\bge-reranker-base\onnx\model_uint8.onnx
_data
    echo     models\bge-reranker-base\tokenizer.json
) else (
    echo   RAG: 跳过
)
echo ========================================
pause