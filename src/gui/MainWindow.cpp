#include "MainWindow.h"
#include <QScrollBar>
#include <QMessageBox>
#include <QTextBlock>
#include <QDateTime>

// ========== WorkerThread 实现 ==========
WorkerThread::WorkerThread(LLMEngine* engine, MemorySystem* memory,
    SkillManager* skillMgr, const std::string& sessionId)
    : m_engine(engine), m_memory(memory), m_skillMgr(skillMgr),
    m_sessionId(sessionId), m_running(true) {
}

void WorkerThread::setPrompt(const std::string& prompt) {
    QMutexLocker locker(&m_mutex);
    m_prompt = prompt;
    m_cond.wakeOne();
}

void WorkerThread::stop() {
    QMutexLocker locker(&m_mutex);
    m_running = false;
    m_cond.wakeOne();
}

void WorkerThread::run() {
    while (m_running) {
        QMutexLocker locker(&m_mutex);
        while (m_prompt.empty() && m_running) {
            m_cond.wait(&m_mutex);
        }
        if (!m_running) break;

        std::string prompt = m_prompt;
        m_prompt.clear();
        locker.unlock();

        std::string accumulatedResponse;

        // 流式生成
        m_engine->generateRaw(
            prompt,
            512,  // maxTokens
            0.7f, // temperature
            [this, &accumulatedResponse](const std::string& token) {
                QString qtoken = QString::fromUtf8(token.c_str());
                accumulatedResponse += token;
                emit tokenReceived(qtoken);
            },
            [this, &accumulatedResponse]() {
                // 处理技能调用
                std::string skillName;
                std::map<std::string, std::string> params;
                if (SkillManager::getInstance().shouldExecuteSkill(accumulatedResponse, skillName, params)) {
                    std::string result = SkillManager::getInstance().executeSkill(skillName, params);
                    emit skillExecuted(QString::fromUtf8(result.c_str()));
                }
                emit finished();
            }
        );
    }
}

// ========== MainWindow 实现 ==========
MainWindow::MainWindow(LLMEngine* engine, MemorySystem* memory,
    SkillManager* skillMgr, const std::string& sessionId,
    QWidget* parent)
    : QMainWindow(parent), m_engine(engine), m_memory(memory),
    m_skillMgr(skillMgr), m_sessionId(sessionId), m_isGenerating(false) {
    setupUI();

    // 启动工作线程
    m_worker = new WorkerThread(engine, memory, skillMgr, sessionId);
    connect(m_worker, &WorkerThread::tokenReceived, this, &MainWindow::onTokenReceived);
    connect(m_worker, &WorkerThread::skillExecuted, this, &MainWindow::onSkillExecuted);
    connect(m_worker, &WorkerThread::finished, this, &MainWindow::onGenerationFinished);
    m_worker->start();

    // 添加欢迎消息
    addMessage("Hello! I am CloseCrab, your AI assistant. I can use skills to help you.", false);
}

MainWindow::~MainWindow() {
    if (m_worker) {
        m_worker->stop();
        m_worker->wait();
        delete m_worker;
    }
}

void MainWindow::setupUI() {
    setWindowTitle("CloseCrab - AI Assistant");
    setMinimumSize(800, 600);

    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

    // 聊天显示区
    m_chatDisplay = new QTextEdit(this);
    m_chatDisplay->setReadOnly(true);
    m_chatDisplay->setFont(QFont("Microsoft YaHei", 10));
    m_chatDisplay->setStyleSheet("QTextEdit { background-color: #f5f5f5; }");
    mainLayout->addWidget(m_chatDisplay);

    // 输入区
    QHBoxLayout* inputLayout = new QHBoxLayout();
    m_inputEdit = new QLineEdit(this);
    m_inputEdit->setPlaceholderText("输入消息... (按 Enter 发送)");
    m_inputEdit->setFont(QFont("Microsoft YaHei", 10));
    connect(m_inputEdit, &QLineEdit::returnPressed, this, &MainWindow::sendMessage);

    m_sendButton = new QPushButton("发送", this);
    connect(m_sendButton, &QPushButton::clicked, this, &MainWindow::sendMessage);

    inputLayout->addWidget(m_inputEdit);
    inputLayout->addWidget(m_sendButton);
    mainLayout->addLayout(inputLayout);

    // 按钮面板（滚动区域）
    QScrollArea* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setMaximumHeight(150);

    m_buttonPanel = new QWidget();
    QGridLayout* buttonLayout = new QGridLayout(m_buttonPanel);
    buttonLayout->setSpacing(5);

    // 技能模式按钮
    QPushButton* btnAuto = new QPushButton("🤖 AUTO 模式", this);
    QPushButton* btnChat = new QPushButton("💬 CHAT 模式", this);
    QPushButton* btnSkillOnly = new QPushButton("⚙️ SKILL 模式", this);
    QPushButton* btnAsk = new QPushButton("❓ ASK 模式", this);

    connect(btnAuto, &QPushButton::clicked, this, &MainWindow::setAutoMode);
    connect(btnChat, &QPushButton::clicked, this, &MainWindow::setChatMode);
    connect(btnSkillOnly, &QPushButton::clicked, this, &MainWindow::setSkillOnlyMode);
    connect(btnAsk, &QPushButton::clicked, this, &MainWindow::setAskMode);

    buttonLayout->addWidget(btnAuto, 0, 0);
    buttonLayout->addWidget(btnChat, 0, 1);
    buttonLayout->addWidget(btnSkillOnly, 0, 2);
    buttonLayout->addWidget(btnAsk, 0, 3);

    // 沙箱模式按钮
    QPushButton* sbDisabled = new QPushButton("🔓 沙箱禁用", this);
    QPushButton* sbAsk = new QPushButton("❓ 沙箱询问", this);
    QPushButton* sbAuto = new QPushButton("🛡️ 沙箱自动", this);
    QPushButton* sbTrusted = new QPushButton("✓ 沙箱信任", this);

    connect(sbDisabled, &QPushButton::clicked, this, &MainWindow::setSandboxDisabled);
    connect(sbAsk, &QPushButton::clicked, this, &MainWindow::setSandboxAsk);
    connect(sbAuto, &QPushButton::clicked, this, &MainWindow::setSandboxAuto);
    connect(sbTrusted, &QPushButton::clicked, this, &MainWindow::setSandboxTrusted);

    buttonLayout->addWidget(sbDisabled, 1, 0);
    buttonLayout->addWidget(sbAsk, 1, 1);
    buttonLayout->addWidget(sbAuto, 1, 2);
    buttonLayout->addWidget(sbTrusted, 1, 3);

    // 其他功能按钮
    QPushButton* btnSkills = new QPushButton("📚 技能列表", this);
    QPushButton* btnClear = new QPushButton("🗑️ 清空历史", this);
    QPushButton* btnNew = new QPushButton("🆕 新会话", this);
    QPushButton* btnHistory = new QPushButton("📜 历史记录", this);

    connect(btnSkills, &QPushButton::clicked, this, &MainWindow::showSkills);
    connect(btnClear, &QPushButton::clicked, this, &MainWindow::clearHistory);
    connect(btnNew, &QPushButton::clicked, this, &MainWindow::newSession);
    connect(btnHistory, &QPushButton::clicked, this, &MainWindow::showHistory);

    buttonLayout->addWidget(btnSkills, 2, 0);
    buttonLayout->addWidget(btnClear, 2, 1);
    buttonLayout->addWidget(btnNew, 2, 2);
    buttonLayout->addWidget(btnHistory, 2, 3);

    scrollArea->setWidget(m_buttonPanel);
    mainLayout->addWidget(scrollArea);

    // 设置样式
    setStyleSheet(R"(
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 11px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QLineEdit {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }
    )");
}

void MainWindow::addMessage(const QString& message, bool isUser) {
    QString timestamp = QDateTime::currentDateTime().toString("hh:mm:ss");
    QString prefix = isUser ? "👤 You" : "🤖 AI";
    QString color = isUser ? "blue" : "green";

    m_chatDisplay->append(QString("<b style='color:%1'>%2 [%3]:</b>").arg(color).arg(prefix).arg(timestamp));
    m_chatDisplay->append(message);
    m_chatDisplay->append("");

    // 滚动到底部
    QScrollBar* bar = m_chatDisplay->verticalScrollBar();
    bar->setValue(bar->maximum());
}

void MainWindow::sendMessage() {
    if (m_isGenerating) {
        addMessage("正在生成中，请稍候...", false);
        return;
    }

    QString input = m_inputEdit->text().trimmed();
    if (input.isEmpty()) return;

    addMessage(input, true);
    m_inputEdit->clear();

    // 构建 prompt（复用你原有的逻辑）
    // 这里简化处理，实际需要根据当前模式构建完整的 systemContent 和历史
    std::string prompt = buildUserMessage(input.toStdString()); // 简化

    m_currentResponse.clear();
    m_isGenerating = true;
    m_worker->setPrompt(prompt);
}

void MainWindow::onTokenReceived(const QString& token) {
    m_currentResponse += token;
    // 更新最后一条消息
    QTextCursor cursor = m_chatDisplay->textCursor();
    cursor.movePosition(QTextCursor::End);
    cursor.select(QTextCursor::BlockUnderCursor);
    if (cursor.selectedText().startsWith("🤖 AI")) {
        cursor.removeSelectedText();
        cursor.insertText(m_currentResponse);
    }
}

void MainWindow::onSkillExecuted(const QString& result) {
    addMessage("[Skill Result]\n" + result, false);
}

void MainWindow::onGenerationFinished() {
    m_isGenerating = false;
    // 保存到记忆
    if (!m_currentResponse.isEmpty()) {
        m_memory->addMemory(m_sessionId, "assistant", m_currentResponse.toStdString());
    }
}

void MainWindow::clearHistory() {
    m_memory->clearMemories(m_sessionId);
    addMessage("对话历史已清空", false);
}

void MainWindow::newSession() {
    // 创建新会话的逻辑
    addMessage("已创建新会话", false);
}

void MainWindow::setAutoMode() {
    m_skillMgr->setMode(SkillMode::AUTO);
    addMessage("技能模式已切换到: AUTO (AI 自动决定)", false);
}

void MainWindow::setChatMode() {
    m_skillMgr->setMode(SkillMode::CHAT_ONLY);
    addMessage("技能模式已切换到: CHAT_ONLY (仅聊天)", false);
}

void MainWindow::setSkillOnlyMode() {
    m_skillMgr->setMode(SkillMode::SKILL_ONLY);
    addMessage("技能模式已切换到: SKILL_ONLY (仅技能)", false);
}

void MainWindow::setAskMode() {
    m_skillMgr->setMode(SkillMode::ASK);
    addMessage("技能模式已切换到: ASK (执行前询问)", false);
}

void MainWindow::showSkills() {
    QString skills = QString::fromStdString(m_skillMgr->getSkillsDescription());
    addMessage("可用技能:\n" + skills, false);
}

void MainWindow::setSandboxDisabled() {
    Sandbox::getInstance().setMode(Sandbox::Mode::DISABLED);
    addMessage("沙箱模式: 禁用 (无保护)", false);
}

void MainWindow::setSandboxAsk() {
    Sandbox::getInstance().setMode(Sandbox::Mode::ASK);
    addMessage("沙箱模式: 询问 (执行前询问)", false);
}

void MainWindow::setSandboxAuto() {
    Sandbox::getInstance().setMode(Sandbox::Mode::AUTO);
    addMessage("沙箱模式: 自动 (自动阻止危险操作)", false);
}

void MainWindow::setSandboxTrusted() {
    Sandbox::getInstance().setMode(Sandbox::Mode::TRUSTED);
    addMessage("沙箱模式: 信任 (仅记录日志)", false);
}