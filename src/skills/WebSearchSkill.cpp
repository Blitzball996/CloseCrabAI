#include "WebSearchSkill.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sstream>

using json = nlohmann::json;

// curl 回调函数（只在 .cpp 中定义）
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append((char*)contents, totalSize);
    return totalSize;
}

std::string WebSearchSkill::execute(const std::map<std::string, std::string>& params) {
    auto it = params.find("query");
    if (it == params.end()) {
        return "错误: 缺少 query 参数";
    }
    return searchWeb(it->second);
}

std::string WebSearchSkill::searchWeb(const std::string& query) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "无法初始化网络请求";
    }

    // URL 编码
    char* encoded = curl_easy_escape(curl, query.c_str(), (int)query.length());
    std::string url = "https://api.duckduckgo.com/?q=" + std::string(encoded) + "&format=json&no_html=1&skip_disambig=1";
    curl_free(encoded);

    spdlog::debug("Searching: {}", query);

    std::string response;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "CloseCrab/1.0");

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        spdlog::error("CURL error: {}", curl_easy_strerror(res));
        return "网络请求失败: " + std::string(curl_easy_strerror(res));
    }

    try {
        json data = json::parse(response);
        std::string result;

        if (data.contains("Abstract") && !data["Abstract"].is_null()) {
            std::string abstract = data["Abstract"].get<std::string>();
            if (!abstract.empty()) {
                result += "📝 " + abstract + "\n\n";
            }
        }

        if (data.contains("Answer") && !data["Answer"].is_null()) {
            std::string answer = data["Answer"].get<std::string>();
            if (!answer.empty()) {
                result += "💡 " + answer + "\n\n";
            }
        }

        if (data.contains("Definition") && !data["Definition"].is_null()) {
            std::string definition = data["Definition"].get<std::string>();
            if (!definition.empty()) {
                result += "📖 " + definition + "\n\n";
            }
        }

        if (data.contains("RelatedTopics") && data["RelatedTopics"].is_array()) {
            int count = 0;
            for (const auto& topic : data["RelatedTopics"]) {
                if (count >= 5) break;
                if (topic.contains("Text")) {
                    result += "🔗 " + topic["Text"].get<std::string>() + "\n";
                    if (topic.contains("FirstURL")) {
                        result += "   " + topic["FirstURL"].get<std::string>() + "\n";
                    }
                    count++;
                }
            }
        }

        if (data.contains("Redirect") && !data["Redirect"].is_null()) {
            result += "\n🔗 更多信息: " + data["Redirect"].get<std::string>();
        }

        if (result.empty()) {
            result = "未找到关于 \"" + query + "\" 的相关信息。";
        }

        return result;

    }
    catch (const std::exception& e) {
        spdlog::error("JSON parse error: {}", e.what());
        return "解析搜索结果失败: " + std::string(e.what());
    }
}