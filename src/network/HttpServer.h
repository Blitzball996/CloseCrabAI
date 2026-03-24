#pragma once
#include <string>
#include <functional>
#include <memory>
#include <thread>
#include <atomic>

class HttpServer {
public:
    HttpServer(int port);
    ~HttpServer();

    void start();
    void stop();

    // 设置聊天回调
    void onChat(std::function<std::string(const std::string& message,
        const std::string& sessionId)> callback);

    bool isRunning() const;
    int getPort() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};