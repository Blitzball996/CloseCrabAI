#include "WordPieceTokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <iostream>
#include <spdlog/spdlog.h>

// ============================================================
// 构造：加载 vocab.txt
// ============================================================
WordPieceTokenizer::WordPieceTokenizer(const std::string& vocabPath,
    int maxInputChars)
    : maxInputChars_(maxInputChars)
{
    std::ifstream fin(vocabPath);
    if (!fin.is_open()) {
        throw std::runtime_error("WordPieceTokenizer: cannot open vocab file: " + vocabPath);
    }

    std::string line;
    int64_t idx = 0;
    while (std::getline(fin, line)) {
        // 去掉尾部 \r（Windows 换行）
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        vocab_[line] = idx;
        idx++;
    }

    // 读取特殊 token id
    auto setSpecial = [&](const std::string& tok, int64_t& dst) {
        auto it = vocab_.find(tok);
        if (it != vocab_.end()) dst = it->second;
        };
    setSpecial("[UNK]", unkId_);
    setSpecial("[CLS]", clsId_);
    setSpecial("[SEP]", sepId_);
    setSpecial("[PAD]", padId_);

    spdlog::info("[Tokenizer] Loaded vocab: {} tokens from {}", vocab_.size(), vocabPath);
}

// ============================================================
// UTF-8 工具
// ============================================================
std::vector<uint32_t> WordPieceTokenizer::utf8ToCodepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len = 1;
        if (c < 0x80) {
            cp = c;
        }
        else if ((c >> 5) == 0x06) {
            cp = c & 0x1F; len = 2;
        }
        else if ((c >> 4) == 0x0E) {
            cp = c & 0x0F; len = 3;
        }
        else if ((c >> 3) == 0x1E) {
            cp = c & 0x07; len = 4;
        }
        else {
            // 无效 UTF-8，跳过
            ++i; continue;
        }
        for (int j = 1; j < len && (i + j) < s.size(); ++j) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3F);
        }
        cps.push_back(cp);
        i += len;
    }
    return cps;
}

std::string WordPieceTokenizer::codepointsToUtf8(const std::vector<uint32_t>& cps) {
    std::string out;
    for (auto cp : cps) {
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
    }
    return out;
}

bool WordPieceTokenizer::isChinese(uint32_t cp) {
    // CJK Unified Ideographs 及其扩展
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
        (cp >= 0x3400 && cp <= 0x4DBF) ||
        (cp >= 0x20000 && cp <= 0x2A6DF) ||
        (cp >= 0x2A700 && cp <= 0x2B73F) ||
        (cp >= 0x2B740 && cp <= 0x2B81F) ||
        (cp >= 0x2B820 && cp <= 0x2CEAF) ||
        (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0x2F800 && cp <= 0x2FA1F) ||
        // 全角标点
        (cp >= 0x3000 && cp <= 0x303F) ||
        (cp >= 0xFF00 && cp <= 0xFFEF);
}

std::string WordPieceTokenizer::tokenizeChinese(const std::string& text) {
    auto cps = utf8ToCodepoints(text);
    std::vector<uint32_t> out;
    for (auto cp : cps) {
        if (isChinese(cp)) {
            out.push_back(' ');
            out.push_back(cp);
            out.push_back(' ');
        }
        else {
            out.push_back(cp);
        }
    }
    return codepointsToUtf8(out);
}

// ============================================================
// Basic tokenization
// ============================================================
std::vector<std::string> WordPieceTokenizer::basicTokenize(const std::string& text) const {
    // 1. 在中文字符两侧加空格
    std::string processed = tokenizeChinese(text);

    // 2. 转小写 + 去除控制字符 + 标点前后加空格
    auto cps = utf8ToCodepoints(processed);
    std::vector<uint32_t> cleaned;
    for (auto cp : cps) {
        if (cp == 0 || cp == 0xFFFD) continue;  // 跳过无效字符
        // 转小写 (仅 ASCII 范围)
        if (cp >= 'A' && cp <= 'Z') cp = cp - 'A' + 'a';
        // 标点前后加空格
        bool isPunct = (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) ||
            (cp >= 0x2000 && cp <= 0x206F);  // 通用标点
        if (isPunct) {
            cleaned.push_back(' ');
            cleaned.push_back(cp);
            cleaned.push_back(' ');
        }
        else {
            cleaned.push_back(cp);
        }
    }

    // 3. 按空格切分
    std::string s = codepointsToUtf8(cleaned);
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) {
        if (!tok.empty()) {
            tokens.push_back(tok);
        }
    }
    return tokens;
}

// ============================================================
// WordPiece 子词切分
// ============================================================
std::vector<std::string> WordPieceTokenizer::wordpieceTokenize(const std::string& word) const {
    auto cps = utf8ToCodepoints(word);
    if (static_cast<int>(cps.size()) > maxInputChars_) {
        return { "[UNK]" };
    }

    std::vector<std::string> subTokens;
    bool isBad = false;
    size_t start = 0;

    while (start < cps.size()) {
        size_t end = cps.size();
        std::string curSubstr;
        bool found = false;

        while (start < end) {
            std::vector<uint32_t> sub(cps.begin() + start, cps.begin() + end);
            std::string s = codepointsToUtf8(sub);
            if (start > 0) {
                s = "##" + s;
            }
            if (vocab_.count(s)) {
                curSubstr = s;
                found = true;
                break;
            }
            end--;
        }

        if (!found) {
            isBad = true;
            break;
        }
        subTokens.push_back(curSubstr);
        start = end;
    }

    if (isBad) {
        return { "[UNK]" };
    }
    return subTokens;
}

// ============================================================
// 公开接口
// ============================================================
std::vector<int64_t> WordPieceTokenizer::tokenize(const std::string& text) const {
    auto words = basicTokenize(text);

    std::vector<int64_t> ids;
    for (const auto& word : words) {
        auto subTokens = wordpieceTokenize(word);
        for (const auto& st : subTokens) {
            auto it = vocab_.find(st);
            ids.push_back(it != vocab_.end() ? it->second : unkId_);
        }
    }
    return ids;
}

std::vector<int64_t> WordPieceTokenizer::tokenizeSingle(const std::string& text) const {
    auto ids = tokenize(text);

    // 截断到 510 (留给 [CLS] + [SEP])
    if (ids.size() > 510) {
        ids.resize(510);
    }

    std::vector<int64_t> result;
    result.reserve(ids.size() + 2);
    result.push_back(clsId_);   // [CLS]
    result.insert(result.end(), ids.begin(), ids.end());
    result.push_back(sepId_);   // [SEP]
    return result;
}

std::vector<int64_t> WordPieceTokenizer::tokenizePair(
    const std::string& query,
    const std::string& doc,
    std::vector<int64_t>& tokenTypeIds) const
{
    auto qIds = tokenize(query);
    auto dIds = tokenize(doc);

    // 截断：query 最多 64 tokens，doc 填满剩余（上限 512 - 3）
    const size_t maxQuery = 64;
    const size_t maxTotal = 509;  // 512 - 3 (CLS + 2*SEP)

    if (qIds.size() > maxQuery) {
        qIds.resize(maxQuery);
    }
    size_t maxDoc = maxTotal - qIds.size();
    if (dIds.size() > maxDoc) {
        dIds.resize(maxDoc);
    }

    // 构建: [CLS] query [SEP] doc [SEP]
    std::vector<int64_t> result;
    result.reserve(qIds.size() + dIds.size() + 3);

    result.push_back(clsId_);
    result.insert(result.end(), qIds.begin(), qIds.end());
    result.push_back(sepId_);
    result.insert(result.end(), dIds.begin(), dIds.end());
    result.push_back(sepId_);

    // token_type_ids: 0 for query part, 1 for doc part
    tokenTypeIds.clear();
    tokenTypeIds.resize(result.size(), 0);
    // doc 部分（第二个 SEP 之后开始算 type=1）
    size_t docStart = 1 + qIds.size() + 1;  // [CLS] + query + [SEP]
    for (size_t i = docStart; i < tokenTypeIds.size(); i++) {
        tokenTypeIds[i] = 1;
    }

    return result;
}