#pragma once
#include <BS_thread_pool.hpp>
#include <future>
#include <spdlog/spdlog.h>

class ThreadPool {
public:
    static ThreadPool& getInstance() {
        static ThreadPool instance;
        return instance;
    }

    // 提交有返回值的任务 - 使用 submit_task
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) {
        // 用 [=] 捕获值，或者把 args 完美转发
        return pool.submit_task([=]() {  // [=] 捕获所有变量的值
            return f(args...);
            });
    }

    // 提交无返回值的任务
    template<typename F, typename... Args>
    void detach(F&& f, Args&&... args) {
        pool.detach_task([&]() {
            f(args...);
            });
    }

    // 等待所有任务完成
    void wait() {
        pool.wait();
    }

    // 获取线程数
    size_t getThreadCount() const {
        return pool.get_thread_count();
    }

private:
    ThreadPool() {
        spdlog::info("ThreadPool created with {} threads", pool.get_thread_count());
    }

    BS::thread_pool<> pool;
};