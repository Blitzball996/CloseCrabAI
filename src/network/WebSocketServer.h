#pragma once
#include <string>
#include <functional>
#include <memory>
#include <map>

class WebSocketServer {
public:
    WebSocketServer(int port);
    ~WebSocketServer();

    void start();
    void stop();
    void onMessage(std::function<void(const std::string& clientId, const std::string& message)> callback);
    void sendMessage(const std::string& clientId, const std::string& message);
    bool isRunning() const;
    int getPort() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};