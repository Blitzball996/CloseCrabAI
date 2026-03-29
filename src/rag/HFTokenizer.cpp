#include "HFTokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <spdlog/spdlog.h>

// 使用 nlohmann/json 解析 tokenizer.json
// 如果你项目里没有，可以用 single-header: https://github.com/nlohmann/json
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// ============================================================
// UTF-8 工具
// ============================================================
static std::vector<uint32_t> utf8ToCodepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        auto c = static_cast<unsigned char>(s[i]);
        int len = 1;
        if (c < 0x80) { cp = c; }
        else if ((c >> 5) == 0x06) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0x0E) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else { ++i; continue; }
        for (int j = 1; j < len && (i + j) < s.size(); ++j)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3F);
        cps.push_back(cp);
        i += len;
    }
    return cps;
}

static std::string codepointToUtf8(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    }
    else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

bool HFTokenizer::isChinese(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
        (cp >= 0x3400 && cp <= 0x4DBF) ||
        (cp >= 0x20000 && cp <= 0x2A6DF) ||
        (cp >= 0x2A700 && cp <= 0x2B73F) ||
        (cp >= 0x2B740 && cp <= 0x2B81F) ||
        (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0x2F800 && cp <= 0x2FA1F) ||
        (cp >= 0x3000 && cp <= 0x303F) ||
        (cp >= 0xFF00 && cp <= 0xFFEF);
}

std::string HFTokenizer::addSpaceAroundChinese(const std::string& text) {
    auto cps = utf8ToCodepoints(text);
    std::string out;
    for (auto cp : cps) {
        if (isChinese(cp)) {
            out += ' ';
            out += codepointToUtf8(cp);
            out += ' ';
        }
        else {
            out += codepointToUtf8(cp);
        }
    }
    return out;
}

std::string HFTokenizer::toLowerAscii(const std::string& s) {
    std::string out = s;
    for (auto& c : out) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
    return out;
}

// ============================================================
// 构造函数：解析 tokenizer.json
// ============================================================
HFTokenizer::HFTokenizer(const std::string& tokenizerJsonPath) {
    std::ifstream f(tokenizerJsonPath);
    if (!f.is_open()) {
        throw std::runtime_error("HFTokenizer: cannot open " + tokenizerJsonPath);
    }

    json j;
    f >> j;

    // 1. 判断模型类型
    std::string modelTypeStr = j["model"]["type"].get<std::string>();
    if (modelTypeStr == "WordPiece") {
        modelType_ = ModelType::WORDPIECE;
    }
    else if (modelTypeStr == "BPE") {
        modelType_ = ModelType::BPE;
    }
    else {
        spdlog::warn("[HFTokenizer] Unknown model type '{}', defaulting to WordPiece", modelTypeStr);
        modelType_ = ModelType::WORDPIECE;
    }

    // 2. 加载词表
    auto& vocabObj = j["model"]["vocab"];
    if (modelType_ == ModelType::WORDPIECE) {
        // WordPiece: vocab 是 { "token": id, ... }
        for (auto it = vocabObj.begin(); it != vocabObj.end(); ++it) {
            vocab_[it.key()] = it.value().get<int64_t>();
        }
    }
    else {
        // BPE: vocab 也是 { "token": id, ... }
        for (auto it = vocabObj.begin(); it != vocabObj.end(); ++it) {
            vocab_[it.key()] = it.value().get<int64_t>();
        }
    }

    // 3. WordPiece: 读取 continuing_subword_prefix
    if (modelType_ == ModelType::WORDPIECE) {
        if (j["model"].contains("continuing_subword_prefix")) {
            continuingSubwordPrefix_ = j["model"]["continuing_subword_prefix"].get<std::string>();
        }
    }

    // 4. BPE: 读取 merges
    if (modelType_ == ModelType::BPE && j["model"].contains("merges")) {
        for (auto& m : j["model"]["merges"]) {
            std::string mergeStr = m.get<std::string>();
            auto spacePos = mergeStr.find(' ');
            if (spacePos != std::string::npos) {
                merges_.emplace_back(
                    mergeStr.substr(0, spacePos),
                    mergeStr.substr(spacePos + 1)
                );
            }
        }
    }

    // 5. 解析 added_tokens 获取特殊 token ID
    if (j.contains("added_tokens")) {
        for (auto& tok : j["added_tokens"]) {
            std::string content = tok["content"].get<std::string>();
            int64_t id = tok["id"].get<int64_t>();

            if (content == "[CLS]" || content == "<s>") clsId_ = id;
            if (content == "[SEP]" || content == "</s>") sepId_ = id;
            if (content == "[UNK]" || content == "<unk>") unkId_ = id;
            if (content == "[PAD]" || content == "<pad>") padId_ = id;
        }
    }

    // 如果 added_tokens 里没找到，尝试从 vocab 里找
    auto tryFind = [&](const std::string& tok, int64_t& dst) {
        if (dst < 0) {
            auto it = vocab_.find(tok);
            if (it != vocab_.end()) dst = it->second;
        }
        };
    tryFind("[CLS]", clsId_); tryFind("<s>", clsId_);
    tryFind("[SEP]", sepId_); tryFind("</s>", sepId_);
    tryFind("[UNK]", unkId_); tryFind("<unk>", unkId_);
    tryFind("[PAD]", padId_); tryFind("<pad>", padId_);

    spdlog::info("[HFTokenizer] Loaded {} (vocab={}, type={})",
        tokenizerJsonPath, vocab_.size(),
        modelType_ == ModelType::WORDPIECE ? "WordPiece" : "BPE");
    spdlog::info("[HFTokenizer] Special tokens: CLS={}, SEP={}, UNK={}, PAD={}",
        clsId_, sepId_, unkId_, padId_);
}

// ============================================================
// Pre-tokenization (分词前预处理)
// ============================================================
std::vector<std::string> HFTokenizer::preTokenize(const std::string& text) const {
    // 中文字符加空格 + 小写化 + 标点分离
    std::string processed = addSpaceAroundChinese(text);
    processed = toLowerAscii(processed);

    // 标点前后加空格
    auto cps = utf8ToCodepoints(processed);
    std::string cleaned;
    for (auto cp : cps) {
        if (cp == 0 || cp == 0xFFFD) continue;
        bool isPunct = (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) ||
            (cp >= 0x2000 && cp <= 0x206F);
        if (isPunct) {
            cleaned += ' ';
            cleaned += codepointToUtf8(cp);
            cleaned += ' ';
        }
        else {
            cleaned += codepointToUtf8(cp);
        }
    }

    // 按空格切分
    std::vector<std::string> words;
    std::istringstream iss(cleaned);
    std::string tok;
    while (iss >> tok) {
        if (!tok.empty()) words.push_back(tok);
    }
    return words;
}

// ============================================================
// WordPiece 子词切分
// ============================================================
std::vector<std::string> HFTokenizer::wordpieceTokenize(const std::string& word) const {
    auto cps = utf8ToCodepoints(word);
    if (cps.size() > 200) return { "[UNK]" };

    std::vector<std::string> subTokens;
    size_t start = 0;

    while (start < cps.size()) {
        size_t end = cps.size();
        std::string found;
        bool matched = false;

        while (start < end) {
            std::string sub;
            for (size_t i = start; i < end; i++)
                sub += codepointToUtf8(cps[i]);
            if (start > 0)
                sub = continuingSubwordPrefix_ + sub;
            if (vocab_.count(sub)) {
                found = sub;
                matched = true;
                break;
            }
            end--;
        }

        if (!matched) return { "[UNK]" };
        subTokens.push_back(found);
        start = end;
    }
    return subTokens;
}

// ============================================================
// BPE 子词切分 (简化版)
// ============================================================
std::vector<std::string> HFTokenizer::bpeTokenize(const std::string& word) const {
    // 初始: 每个 UTF-8 字符作为一个 token
    auto cps = utf8ToCodepoints(word);
    std::vector<std::string> symbols;
    for (auto cp : cps) {
        symbols.push_back(codepointToUtf8(cp));
    }

    // 应用 merge 规则
    for (const auto& [first, second] : merges_) {
        std::vector<std::string> newSymbols;
        size_t i = 0;
        while (i < symbols.size()) {
            if (i + 1 < symbols.size() && symbols[i] == first && symbols[i + 1] == second) {
                newSymbols.push_back(first + second);
                i += 2;
            }
            else {
                newSymbols.push_back(symbols[i]);
                i++;
            }
        }
        symbols = std::move(newSymbols);
        if (symbols.size() == 1) break;
    }

    return symbols;
}

// ============================================================
// token → id
// ============================================================
int64_t HFTokenizer::tokenToId(const std::string& token) const {
    auto it = vocab_.find(token);
    return (it != vocab_.end()) ? it->second : unkId_;
}

// ============================================================
// 完整分词（不含特殊 token）
// ============================================================
std::vector<int64_t> HFTokenizer::tokenize(const std::string& text) const {
    auto words = preTokenize(text);
    std::vector<int64_t> ids;

    for (const auto& word : words) {
        std::vector<std::string> subTokens;
        if (modelType_ == ModelType::WORDPIECE) {
            subTokens = wordpieceTokenize(word);
        }
        else {
            subTokens = bpeTokenize(word);
        }
        for (const auto& st : subTokens) {
            ids.push_back(tokenToId(st));
        }
    }
    return ids;
}

// ============================================================
// 单句编码
// ============================================================
std::vector<int64_t> HFTokenizer::encodeSingle(const std::string& text) const {
    auto ids = tokenize(text);

    // 截断到 510
    if (ids.size() > 510) ids.resize(510);

    std::vector<int64_t> result;
    result.reserve(ids.size() + 2);
    result.push_back(clsId_);
    result.insert(result.end(), ids.begin(), ids.end());
    result.push_back(sepId_);
    return result;
}

// ============================================================
// 句对编码
// ============================================================
std::vector<int64_t> HFTokenizer::encodePair(
    const std::string& textA,
    const std::string& textB,
    std::vector<int64_t>& tokenTypeIds) const
{
    auto aIds = tokenize(textA);
    auto dIds = tokenize(textB);

    // 截断
    const size_t maxA = 64;
    if (aIds.size() > maxA) aIds.resize(maxA);

    if (modelType_ == ModelType::WORDPIECE) {
        // BERT: [CLS] A [SEP] B [SEP]  →  3 个特殊 token
        size_t maxB = 509 - aIds.size();  // 512 - 3
        if (dIds.size() > maxB) dIds.resize(maxB);

        std::vector<int64_t> result;
        result.reserve(aIds.size() + dIds.size() + 3);
        result.push_back(clsId_);
        result.insert(result.end(), aIds.begin(), aIds.end());
        result.push_back(sepId_);
        result.insert(result.end(), dIds.begin(), dIds.end());
        result.push_back(sepId_);

        // token_type_ids: 0 for A part (CLS + A + SEP), 1 for B part (B + SEP)
        tokenTypeIds.assign(result.size(), 0);
        size_t bStart = 1 + aIds.size() + 1;
        for (size_t i = bStart; i < tokenTypeIds.size(); i++)
            tokenTypeIds[i] = 1;

        return result;
    }
    else {
        // XLM-R: <s> A </s></s> B </s>  →  4 个特殊 token
        size_t maxB = 508 - aIds.size();  // 512 - 4
        if (dIds.size() > maxB) dIds.resize(maxB);

        std::vector<int64_t> result;
        result.reserve(aIds.size() + dIds.size() + 4);
        result.push_back(clsId_);   // <s>
        result.insert(result.end(), aIds.begin(), aIds.end());
        result.push_back(sepId_);   // </s>
        result.push_back(sepId_);   // </s>  (XLM-R pair separator)
        result.insert(result.end(), dIds.begin(), dIds.end());
        result.push_back(sepId_);   // </s>

        // XLM-R 不真正使用 token_type_ids，全 0
        tokenTypeIds.assign(result.size(), 0);
        return result;
    }
}