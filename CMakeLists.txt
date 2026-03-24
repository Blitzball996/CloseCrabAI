cmake_minimum_required(VERSION 3.20)
project(CloseCrab VERSION 1.0.0 LANGUAGES C CXX)

if(MSVC)
    add_compile_options(/Zc:char8_t-)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ============================================
# 依赖库
# ============================================

# ============================================
# libcurl - 网络请求库
# ============================================
set(CURL_LIBRARY "G:/vcpkg/installed/x64-windows-static/lib/libcurl.lib")
set(CURL_INCLUDE_DIR "G:/vcpkg/installed/x64-windows-static/include")

if(EXISTS ${CURL_LIBRARY})
    message(STATUS "curl.lib found: ${CURL_LIBRARY}")
    set(CURL_LIBRARIES ${CURL_LIBRARY})
    set(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIR})
else()
    message(FATAL_ERROR "curl.lib not found!")
endif()

# 查找 Qt6 (如果失败则尝试 Qt5)
#find_package(Qt6 COMPONENTS Core Widgets Network REQUIRED)
#if(NOT Qt6_FOUND)
#    find_package(Qt5 COMPONENTS Core Widgets Network REQUIRED)
#    set(QT_LIBRARIES Qt5::Core Qt5::Widgets Qt5::Network)
#else()
#    set(QT_LIBRARIES Qt6::Core Qt6::Widgets Qt6::Network)
#endif()


# spdlog
FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git GIT_TAG v1.14.1)
FetchContent_MakeAvailable(spdlog)

# json
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

# yaml-cpp
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git GIT_TAG 0.8.0)
FetchContent_MakeAvailable(yaml-cpp)

# CLI11
FetchContent_Declare(CLI11 GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.4.2)
FetchContent_MakeAvailable(CLI11)

# thread_pool
FetchContent_Declare(thread_pool GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git GIT_TAG v5.1.0)
FetchContent_MakeAvailable(thread_pool)

# ============================================
# SQLite3 - 修复版
# ============================================
FetchContent_Declare(sqlite3 URL https://www.sqlite.org/2024/sqlite-autoconf-3460100.tar.gz)
FetchContent_MakeAvailable(sqlite3)

# ============================================
# sqlite-vec - 向量数据库扩展
# ============================================
FetchContent_Declare(
    sqlite_vec
    GIT_REPOSITORY https://github.com/asg017/sqlite-vec.git
    GIT_TAG v0.1.6
)
FetchContent_MakeAvailable(sqlite_vec)
# 获取 sqlite-vec 源码路径
set(SQLITE_VEC_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/sqlite_vec-src")

# 检查 sqlite3 是否作为目标存在
if(TARGET sqlite3)
    set(SQLITE3_LIBRARIES sqlite3)
    set(SQLITE3_INCLUDE_DIRS ${sqlite3_SOURCE_DIR})
    message(STATUS "SQLite3 target found")
else()
    # 手动添加编译
    add_library(sqlite3 STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)
    target_include_directories(sqlite3 PUBLIC ${sqlite3_SOURCE_DIR})
    set(SQLITE3_LIBRARIES sqlite3)
    set(SQLITE3_INCLUDE_DIRS ${sqlite3_SOURCE_DIR})
    message(STATUS "SQLite3 compiled manually")
endif()

# sqlite3 本身就是一个库目标
set(SQLITE3_LIBRARIES sqlite3)
set(SQLITE3_INCLUDE_DIRS ${sqlite3_SOURCE_DIR})

# ============================================
# OpenSSL
# ============================================
set(OPENSSL_ROOT_DIR "C:/Program Files/OpenSSL-Win64")
set(OPENSSL_INCLUDE_DIR "${OPENSSL_ROOT_DIR}/include")
set(OPENSSL_LIBRARY_DIR "${OPENSSL_ROOT_DIR}/lib/VC/x64/MD")

if(EXISTS ${OPENSSL_LIBRARY_DIR}/libssl.lib AND EXISTS ${OPENSSL_LIBRARY_DIR}/libcrypto.lib)
    add_library(OpenSSL_SSL STATIC IMPORTED)
    set_target_properties(OpenSSL_SSL PROPERTIES
        IMPORTED_LOCATION "${OPENSSL_LIBRARY_DIR}/libssl.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_INCLUDE_DIR}"
    )
    add_library(OpenSSL_Crypto STATIC IMPORTED)
    set_target_properties(OpenSSL_Crypto PROPERTIES
        IMPORTED_LOCATION "${OPENSSL_LIBRARY_DIR}/libcrypto.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_INCLUDE_DIR}"
    )
    message(STATUS "OpenSSL found: ${OPENSSL_LIBRARY_DIR}")
    set(OPENSSL_FOUND TRUE)
else()
    message(WARNING "OpenSSL not found")
    set(OPENSSL_FOUND FALSE)
endif()

# ============================================
# ZLIB
# ============================================
set(ZLIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/zlib")
set(ZLIB_LIBRARY "${ZLIB_PATH}/build/Release/z.lib")
set(ZLIB_INCLUDE_DIR "${ZLIB_PATH}")

if(EXISTS ${ZLIB_LIBRARY})
    add_library(ZLIB::ZLIB STATIC IMPORTED)
    set_target_properties(ZLIB::ZLIB PROPERTIES
        IMPORTED_LOCATION "${ZLIB_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR}"
    )
    message(STATUS "ZLIB found: ${ZLIB_LIBRARY}")
    set(ZLIB_FOUND TRUE)
else()
    message(WARNING "ZLIB not found")
    set(ZLIB_FOUND FALSE)
endif()

# ============================================
# ixwebsocket
# ============================================
set(IXWEBSOCKET_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/ixwebsocket")

if(EXISTS ${IXWEBSOCKET_PATH}/CMakeLists.txt)
    add_subdirectory(${IXWEBSOCKET_PATH})
    message(STATUS "ixwebsocket found")
    set(IXWEBSOCKET_FOUND TRUE)
else()
    message(WARNING "ixwebsocket not found")
    set(IXWEBSOCKET_FOUND FALSE)
endif()

# ============================================
# llama.cpp
# ============================================
set(LLAMA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/llama.cpp")
set(LLAMA_BUILD_PATH "${LLAMA_PATH}/build/src/Release")

if(EXISTS ${LLAMA_BUILD_PATH}/llama.lib)
    add_library(llama STATIC IMPORTED)
    set_target_properties(llama PROPERTIES
        IMPORTED_LOCATION "${LLAMA_BUILD_PATH}/llama.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_PATH}/include"
    )
    message(STATUS "Using CUDA llama.lib from: ${LLAMA_BUILD_PATH}")
else()
    message(FATAL_ERROR "CUDA llama.lib not found!")
endif()

# ============================================
# httplib - HTTP 服务器库
# ============================================
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.1
)
FetchContent_MakeAvailable(httplib)

# ============================================
# 源文件
# ============================================
set(SOURCES
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
# 可执行文件
# ============================================
add_executable(closecrab ${SOURCES})

target_link_libraries(closecrab PRIVATE 
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    yaml-cpp::yaml-cpp
    CLI11::CLI11
    ${SQLITE3_LIBRARIES}
    llama
    ${CURL_LIBRARIES}
    ws2_32
    wldap32
    crypt32           
    advapi32          
    user32            
    iphlpapi
    secur32
    #sqlite-vec
    #${QT_LIBRARIES}
)

if(OPENSSL_FOUND AND IXWEBSOCKET_FOUND)
    target_link_libraries(closecrab PRIVATE ixwebsocket OpenSSL_SSL OpenSSL_Crypto)
    target_include_directories(closecrab PRIVATE ${IXWEBSOCKET_PATH})
endif()

if(ZLIB_FOUND)
    target_link_libraries(closecrab PRIVATE ZLIB::ZLIB)
endif()

target_include_directories(closecrab
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${thread_pool_SOURCE_DIR}/include
    ${LLAMA_PATH}
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
    ${SQLITE3_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${httplib_SOURCE_DIR}
    ${CURL_INCLUDE_DIRS}
    #${CURL_INCLUDE_DIRS}
    #${SQLITE_VEC_SOURCE_DIR}
)

# ============================================
# 输出配置信息
# ============================================
message(STATUS "========================================")
message(STATUS "CloseCrab v${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenSSL: ${OPENSSL_FOUND}")
message(STATUS "ixwebsocket: ${IXWEBSOCKET_FOUND}")
message(STATUS "ZLIB: ${ZLIB_FOUND}")
message(STATUS "libcurl: ${CURL_VERSION_STRING}")
message(STATUS "========================================")