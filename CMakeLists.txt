cmake_minimum_required(VERSION 3.20)

project(CloseCrab VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_PREFIX_PATH "G:/vcpkg/installed/x64-windows")

# ============================================
# ✅ 全局配置
# ============================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/Zc:char8_t-)

    # ✅ 全部统一 /MD
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# ============================================
# CUDA
# ============================================

find_package(CUDAToolkit REQUIRED)

# ============================================
# vcpkg 依赖
# ============================================

find_package(CURL CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

# ============================================
# FetchContent
# ============================================

include(FetchContent)

# spdlog
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1)
FetchContent_MakeAvailable(spdlog)

# json
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

# yaml-cpp
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0)
FetchContent_MakeAvailable(yaml-cpp)

# CLI11
FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.4.2)
FetchContent_MakeAvailable(CLI11)

# thread pool
FetchContent_Declare(thread_pool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG v5.1.0)
FetchContent_MakeAvailable(thread_pool)

# sqlite3
FetchContent_Declare(sqlite3
    URL https://www.sqlite.org/2024/sqlite-autoconf-3460100.tar.gz)
FetchContent_MakeAvailable(sqlite3)

if(NOT TARGET sqlite3)
    add_library(sqlite3 STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3 PUBLIC ${sqlite3_SOURCE_DIR})
endif()

# httplib（header-only）
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.1)
FetchContent_MakeAvailable(httplib)

# ============================================
# ixwebsocket（本地）
# ============================================

set(IXWEBSOCKET_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/ixwebsocket")

if(EXISTS ${IXWEBSOCKET_PATH}/CMakeLists.txt)
    add_subdirectory(${IXWEBSOCKET_PATH})
else()
    message(FATAL_ERROR "ixwebsocket not found in external/")
endif()

# ============================================
# FAISS（vcpkg 动态）
# ============================================

set(FAISS_DIR "G:/vcpkg/installed/x64-windows")

add_library(faiss SHARED IMPORTED)
set_target_properties(faiss PROPERTIES
    IMPORTED_IMPLIB "${FAISS_DIR}/lib/faiss.lib"
    IMPORTED_LOCATION "${FAISS_DIR}/bin/faiss.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${FAISS_DIR}/include"
)

target_link_libraries(faiss INTERFACE
    "${FAISS_DIR}/lib/openblas.lib"
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
    CUDA::curand
)

# ============================================
# llama.cpp（必须 /MD 编译）
# ============================================

set(LLAMA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")
set(LLAMA_LIB_DIR "${LLAMA_PATH}/build/src/Release")
set(GGML_LIB_DIR "${LLAMA_PATH}/build/ggml/src/Release")

add_library(llama STATIC IMPORTED)
set_target_properties(llama PROPERTIES
    IMPORTED_LOCATION "${LLAMA_LIB_DIR}/llama.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_PATH}/include"
)

add_library(ggml STATIC IMPORTED)
set_target_properties(ggml PROPERTIES
    IMPORTED_LOCATION "${GGML_LIB_DIR}/ggml.lib"
)

add_library(ggml-base STATIC IMPORTED)
set_target_properties(ggml-base PROPERTIES
    IMPORTED_LOCATION "${GGML_LIB_DIR}/ggml-base.lib"
)

add_library(ggml-cpu STATIC IMPORTED)
set_target_properties(ggml-cpu PROPERTIES
    IMPORTED_LOCATION "${GGML_LIB_DIR}/ggml-cpu.lib"
)

# ============================================
# 可执行文件
# ============================================

add_executable(closecrab
    src/main_interactive.cpp
    src/config/Config.cpp
    src/core/SessionManager.cpp
    src/core/ThreadPool.cpp
    src/memory/MemorySystem.cpp
    src/llm/LLMEngine.cpp
    src/network/WebSocketServer.cpp
    src/network/HttpServer.cpp
    src/skills/SkillManager.cpp
    src/skills/Skill.cpp
    src/skills/WebSearchSkill.cpp
    src/security/Sandbox.cpp
    src/rag/RAGManager.cpp
)

# ============================================
# 链接（核心）
# ============================================

# 设置 FAISS_GPU_ENABLED 宏
add_definitions(-DFAISS_GPU_ENABLED)

target_link_libraries(closecrab PRIVATE
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    yaml-cpp
    CLI11::CLI11
    sqlite3

    faiss
    llama
    ggml
    ggml-base
    ggml-cpu

    ixwebsocket

    CURL::libcurl
    ZLIB::ZLIB

    ws2_32
    wldap32
    crypt32
    advapi32
    user32
    iphlpapi
    secur32
)

# ============================================
# include
# ============================================

target_include_directories(closecrab PRIVATE
    src
    ${thread_pool_SOURCE_DIR}/include
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
    ${FAISS_DIR}/include
    ${IXWEBSOCKET_PATH}
    ${httplib_SOURCE_DIR}
)

# ============================================
# ✅ 自动复制 DLL（关键！！！）
# ============================================

add_custom_command(TARGET closecrab POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${FAISS_DIR}/bin/faiss.dll"
    $<TARGET_FILE_DIR:closecrab>
)

# ============================================
# 输出
# ============================================

message(STATUS "========================================")
message(STATUS "CloseCrab v${PROJECT_VERSION}")
message(STATUS "MSVC Runtime: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
message(STATUS "========================================")