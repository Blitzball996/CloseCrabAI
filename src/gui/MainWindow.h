#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTextEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QScrollArea>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>

#include "llm/LLMEngine.h"
#include "memory/MemorySystem.h"
#include "skills/SkillManager.h"

class WorkerThread : public QThread {
    Q_OBJECT
public:
    WorkerThread(LLMEngine* engine, MemorySystem* memory,
        SkillManager* skillMgr, const std::string& sessionId);
    void setPrompt(const std::string& prompt);
    void stop();

signals:
    void tokenReceived(const QString& token);
    void finished();
    void skillExecuted(const QString& result);

protected:
    void run() override;

private:
    LLMEngine* m_engine;
    MemorySystem* m_memory;
    SkillManager* m_skillMgr;
    std::string m_sessionId;
    std::string m_prompt;
    bool m_running;
    QMutex m_mutex;
    QWaitCondition m_cond;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(LLMEngine* engine, MemorySystem* memory,
        SkillManager* skillMgr, const std::string& sessionId,
        QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void sendMessage();
    void onTokenReceived(const QString& token);
    void onSkillExecuted(const QString& result);
    void onGenerationFinished();
    void clearHistory();
    void newSession();
    void setAutoMode();
    void setChatMode();
    void setSkillOnlyMode();
    void setAskMode();
    void showSkills();
    void setSandboxDisabled();
    void setSandboxAsk();
    void setSandboxAuto();
    void setSandboxTrusted();

private:
    void setupUI();
    void addMessage(const QString& message, bool isUser);

    QTextEdit* m_chatDisplay;
    QLineEdit* m_inputEdit;
    QPushButton* m_sendButton;
    QWidget* m_buttonPanel;

    LLMEngine* m_engine;
    MemorySystem* m_memory;
    SkillManager* m_skillMgr;
    std::string m_sessionId;

    WorkerThread* m_worker;
    QString m_currentResponse;
    bool m_isGenerating;
};

#endif