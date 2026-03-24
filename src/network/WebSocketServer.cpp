#include "WebSocketServer.h"
#include <spdlog/spdlog.h>
#include <ixwebsocket/IXWebSocketServer.h>
#include <thread>
#include <atomic>
#include <map>
#include <mutex>
#include <system_error>
#include <winsock2.h>

class WebSocketServer::Impl {
public:
    int port;
    std::unique_ptr<ix::WebSocketServer> server;
    std::atomic<bool> running{ false };
    std::thread serverThread;
    std::function<void(const std::string&, const std::string&)> callback;
    std::map<std::string, std::shared_ptr<ix::WebSocket>> clients;
    std::mutex clientsMutex;

    Impl(int p) : port(p) {}
};

WebSocketServer::WebSocketServer(int port) : pImpl(std::make_unique<Impl>(port)) {}

WebSocketServer::~WebSocketServer() {
    stop();
}

void WebSocketServer::onMessage(std::function<void(const std::string&, const std::string&)> callback) {
    pImpl->callback = callback;
}

void WebSocketServer::start() {
    if (pImpl->running) return;

    try {
        pImpl->server = std::make_unique<ix::WebSocketServer>(pImpl->port);

        ix::WebSocketServer::OnConnectionCallback onConnection =
            [this](std::weak_ptr<ix::WebSocket> weakWebSocket,
                std::shared_ptr<ix::ConnectionState> connectionState)
            {
                if (auto webSocket = weakWebSocket.lock()) {
                    std::string clientId = std::to_string(reinterpret_cast<uintptr_t>(webSocket.get()));

                    {
                        std::lock_guard<std::mutex> lock(pImpl->clientsMutex);
                        pImpl->clients[clientId] = webSocket;
                    }

                    spdlog::info("WebSocket client connected: {}", clientId);

                    webSocket->setOnMessageCallback(
                        [this, clientId](const ix::WebSocketMessagePtr& msg) {
                            if (msg->type == ix::WebSocketMessageType::Message) {
                                std::string message = msg->str;
                                spdlog::debug("WebSocket message from {}: {}", clientId, message);
                                if (pImpl->callback) {
                                    pImpl->callback(clientId, message);
                                }
                            }
                            else if (msg->type == ix::WebSocketMessageType::Close) {
                                spdlog::info("WebSocket client disconnected: {}", clientId);
                                std::lock_guard<std::mutex> lock(pImpl->clientsMutex);
                                pImpl->clients.erase(clientId);
                            }
                        }
                    );
                }
            };

        pImpl->server->setOnConnectionCallback(onConnection);

        pImpl->running = true;
        pImpl->serverThread = std::thread([this]() {
            spdlog::info("Attempting to listen on port {}", pImpl->port);

            auto res = pImpl->server->listen();
            if (!res.first) {
                // 获取详细错误信息
                int errorCode = WSAGetLastError();
                spdlog::error("========================================");
                spdlog::error("WebSocket server failed to start");
                spdlog::error("  Error message: {}", res.second);
                spdlog::error("  Windows error code: {}", errorCode);

                // 分析错误原因
                if (errorCode == WSAEADDRINUSE) {
                    spdlog::error("  Cause: Port {} is already in use", pImpl->port);
                    spdlog::error("  Solution: Use a different port with --ws-port <port>");
                }
                else if (errorCode == WSAEACCES) {
                    spdlog::error("  Cause: Permission denied");
                    spdlog::error("  Solution: Run as Administrator");
                }
                else if (errorCode == WSAEADDRNOTAVAIL) {
                    spdlog::error("  Cause: Address not available");
                    spdlog::error("  Solution: Check if the address is valid");
                }
                else {
                    spdlog::error("  Cause: Unknown error");
                }
                spdlog::error("========================================");
                spdlog::error("WebSocket server will be disabled. Use --no-ws to suppress this message.");

                pImpl->running = false;
                return;
            }

            spdlog::info("WebSocket server started on ws://localhost:{}", pImpl->port);
            pImpl->server->start();
            });

    }
    catch (const std::exception& e) {
        spdlog::error("Exception creating WebSocket server: {}", e.what());
        pImpl->running = false;
    }
}

void WebSocketServer::stop() {
    if (!pImpl->running) return;

    if (pImpl->server) {
        pImpl->server->stop();
    }

    if (pImpl->serverThread.joinable()) {
        pImpl->serverThread.join();
    }

    pImpl->running = false;

    // 清理 Winsock
    WSACleanup();

    spdlog::info("WebSocket server stopped");
}

void WebSocketServer::sendMessage(const std::string& clientId, const std::string& message) {
    std::lock_guard<std::mutex> lock(pImpl->clientsMutex);
    auto it = pImpl->clients.find(clientId);
    if (it != pImpl->clients.end()) {
        it->second->send(message);
        spdlog::debug("Sent message to {}: {}", clientId, message.substr(0, 50));
    }
    else {
        spdlog::warn("Client {} not found", clientId);
    }
}

bool WebSocketServer::isRunning() const {
    return pImpl->running;
}

int WebSocketServer::getPort() const {
    return pImpl->port;
}