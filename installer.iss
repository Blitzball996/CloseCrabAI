; CloseCrab 安装脚本
#define MyAppName "CloseCrab"
#define MyAppVersion "1.0.12"
#define MyAppPublisher "Blitzball Inc."
#define MyAppExeName "closecrab.exe"

[Setup]
AppId={{CloseCrab-AI-Engine}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
OutputDir=installer
OutputBaseFilename=CloseCrab_Setup
PrivilegesRequired=lowest
WizardStyle=modern

; ===== 图标配置 =====
SetupIconFile=icons\closecrab.ico

[Files]
; 主程序
Source: "out\build\x64-Release\closecrab.exe"; DestDir: "{app}"; Flags: ignoreversion
; DLL 文件
Source: "out\build\x64-Release\*.dll"; DestDir: "{app}"; Flags: ignoreversion
; 配置文件
Source: "config\config.yaml"; DestDir: "{app}\config"; Flags: ignoreversion
; 模型下载脚本
Source: "download_model.bat"; DestDir: "{app}"; Flags: ignoreversion
; 启动脚本
Source: "run.bat"; DestDir: "{app}"; Flags: ignoreversion

; ===== 图标文件 =====
Source: "icons\closecrab.ico"; DestDir: "{app}\icons"; Flags: ignoreversion

[Icons]
; 开始菜单
Name: "{group}\{#MyAppName}"; Filename: "{app}\run.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0
Name: "{group}\模型下载器"; Filename: "{app}\download_model.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; 桌面快捷方式
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\run.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
; 安装完成后运行模型下载器
Filename: "{app}\download_model.bat"; Description: "下载 AI 模型 (推荐)"; Flags: postinstall runhidden; Check: ShouldDownloadModel

[Code]
var
  ModelPage: TWizardPage;
  ModelCombo: TNewComboBox;
  ModelDescLabel: TNewStaticText;
  RAGCheckBox: TNewCheckBox;

function ShouldDownloadModel: Boolean;
begin
  Result := True;
end;

procedure InitializeWizard;
var
  Label1, Label2, RAGLabel: TNewStaticText;
begin
  { 创建自定义页面 }
  ModelPage := CreateCustomPage(wpSelectTasks, '选择 AI 模型', '请选择要下载的模型');
  
  Label1 := TNewStaticText.Create(ModelPage);
  Label1.Parent := ModelPage.Surface;
  Label1.Caption := 'CloseCrab 需要下载模型文件才能运行。';
  Label1.AutoSize := True;
  Label1.Top := 0;
  
  Label2 := TNewStaticText.Create(ModelPage);
  Label2.Parent := ModelPage.Surface;
  Label2.Caption := '请选择 LLM 大语言模型：' + #13#10 + 
                   ' - Qwen2.5-7B: 4.5GB，推荐日常使用' + #13#10 +
                   ' - Qwen2.5-14B: 8.5GB，更强能力' + #13#10 +
                   ' - Qwen2.5-3B: 2GB，轻量快速' + #13#10 +
                   ' - Qwen2.5-1.5B: 1.2GB，极速响应';
  Label2.AutoSize := True;
  Label2.Top := 25;
  
  ModelCombo := TNewComboBox.Create(ModelPage);
  ModelCombo.Parent := ModelPage.Surface;
  ModelCombo.Top := 110;
  ModelCombo.Width := 350;
  ModelCombo.Style := csDropDownList;
  ModelCombo.Items.Add('Qwen2.5-7B (推荐, 4.5GB, 8GB显存)');
  ModelCombo.Items.Add('Qwen2.5-14B (更强, 8.5GB, 12GB显存)');
  ModelCombo.Items.Add('Qwen2.5-3B (轻量, 2GB, 6GB显存)');
  ModelCombo.Items.Add('Qwen2.5-1.5B (极速, 1.2GB, 4GB显存)');
  ModelCombo.ItemIndex := 0;

  { ---- RAG 模型选项（新增） ---- }
  RAGLabel := TNewStaticText.Create(ModelPage);
  RAGLabel.Parent := ModelPage.Surface;
  RAGLabel.Caption := 'RAG 知识库检索模型（提升回答质量）：';
  RAGLabel.AutoSize := True;
  RAGLabel.Top := 150;
  RAGLabel.Font.Style := [fsBold];

  RAGCheckBox := TNewCheckBox.Create(ModelPage);
  RAGCheckBox.Parent := ModelPage.Surface;
  RAGCheckBox.Top := 172;
  RAGCheckBox.Width := 400;
  RAGCheckBox.Caption := '下载 RAG 模型 (Embedding 134MB + Reranker 1.1GB)';
  RAGCheckBox.Checked := True;

  ModelDescLabel := TNewStaticText.Create(ModelPage);
  ModelDescLabel.Parent := ModelPage.Surface;
  ModelDescLabel.Caption := '  - Embedding: BAAI/bge-small-zh-v1.5（中文语义向量）' + #13#10 +
                            '  - Reranker: BAAI/bge-reranker-base（精排重排序）' + #13#10 +
                            '提示：模型下载需要几分钟到几小时，取决于网速。';
  ModelDescLabel.AutoSize := True;
  ModelDescLabel.Top := 198;
end;

function GetSelectedModelUrl: string;
begin
  case ModelCombo.ItemIndex of
    0: Result := 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf';
    1: Result := 'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf';
    2: Result := 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf';
    3: Result := 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf';
  else
    Result := 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf';
  end;
end;

function GetSelectedModelName: string;
begin
  case ModelCombo.ItemIndex of
    0: Result := 'qwen2.5-7b-instruct-q4_k_m.gguf';
    1: Result := 'qwen2.5-14b-instruct-q4_k_m.gguf';
    2: Result := 'qwen2.5-3b-instruct-q4_k_m.gguf';
    3: Result := 'qwen2.5-1.5b-instruct-q4_k_m.gguf';
  else
    Result := 'qwen2.5-7b-instruct-q4_k_m.gguf';
  end;
end;

function UpdateConfigFile(ConfigPath, ModelName: string): Boolean;
var
  Lines: TStringList;
  i: Integer;
  Modified, HasRag: Boolean;
begin
  Result := False;
  Lines := TStringList.Create;
  try
    if FileExists(ConfigPath) then
      Lines.LoadFromFile(ConfigPath)
    else
      Lines.Text := '# CloseCrab 配置文件'#13#10 +
                    'server:'#13#10 +
                    '  port: 9001'#13#10 +
                    '  host: "127.0.0.1"'#13#10 +
                    'database:'#13#10 +
                    '  path: "data/closecrab.db"'#13#10 +
                    'logging:'#13#10 +
                    '  level: "info"'#13#10 +
                    'llm:'#13#10 +
                    '  model_path: "models/' + ModelName + '"'#13#10 +
                    '  max_tokens: 512'#13#10 +
                    '  temperature: 0.7'#13#10 +
                    'rag:'#13#10 +
                    '  embedding_model_path: "models/bge-small-zh/model.onnx"'#13#10 +
                    '  embedding_vocab_path: "models/bge-small-zh/vocab.txt"'#13#10 +
                    '  reranker_model_path: "models/bge-reranker-base/model.onnx"'#13#10 +
                    '  reranker_vocab_path: "models/bge-reranker-base/vocab.txt"';
    
    Modified := False;
    HasRag := False;
    for i := 0 to Lines.Count - 1 do
    begin
      if (Pos('model_path:', Lines[i]) > 0) and (Pos('embedding', Lines[i]) = 0) and (Pos('reranker', Lines[i]) = 0) then
      begin
        Lines[i] := '  model_path: "models/' + ModelName + '"';
        Modified := True;
      end;
      if Pos('rag:', Lines[i]) > 0 then
        HasRag := True;
    end;
    
    if not Modified then
    begin
      Lines.Add('llm:');
      Lines.Add('  model_path: "models/' + ModelName + '"');
      Lines.Add('  max_tokens: 512');
      Lines.Add('  temperature: 0.7');
    end;

    { 如果没有 rag 段，追加 }
    if not HasRag then
    begin
      Lines.Add('');
      Lines.Add('rag:');
      Lines.Add('  embedding_model_path: "models/bge-small-zh/model.onnx"');
      Lines.Add('  embedding_vocab_path: "models/bge-small-zh/vocab.txt"');
      Lines.Add('  reranker_model_path: "models/bge-reranker-base/model.onnx"');
      Lines.Add('  reranker_vocab_path: "models/bge-reranker-base/vocab.txt"');
    end;
    
    Lines.SaveToFile(ConfigPath);
    Result := True;
  finally
    Lines.Free;
  end;
end;

{ 下载单个文件，返回是否成功 }
function DownloadFile(Url, DestPath: string): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('curl', '-L -o "' + DestPath + '" "' + Url + '"',
                 '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := Result and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ModelUrl, ModelName, ConfigPath: string;
  ModelDir, EmbDir, RerDir: string;
  DownloadOK: Boolean;
  FailedFiles: string;
begin
  if CurStep = ssPostInstall then
  begin
    ModelUrl := GetSelectedModelUrl;
    ModelName := GetSelectedModelName;
    ConfigPath := ExpandConstant('{app}\config\config.yaml');
    
    ModelDir := ExpandConstant('{app}\models');
    CreateDir(ModelDir);
    
    { 更新配置文件 }
    UpdateConfigFile(ConfigPath, ModelName);
    
    { ---- 下载 LLM ---- }
    if not DownloadFile(ModelUrl, ModelDir + '\' + ModelName) then
    begin
      MsgBox('LLM 模型下载失败，请稍后手动运行 download_model.bat 重试。' + #13#10 +
             '下载地址: ' + ModelUrl, mbError, MB_OK);
    end
    else
    begin
      MsgBox('LLM 模型下载完成！' + #13#10 +
             '文件: ' + ModelDir + '\' + ModelName, mbInformation, MB_OK);
    end;

    { ---- 下载 RAG 模型（如果勾选） ---- }
    if RAGCheckBox.Checked then
    begin
      EmbDir := ModelDir + '\bge-small-zh';
      RerDir := ModelDir + '\bge-reranker-base';
      CreateDir(EmbDir);
      CreateDir(RerDir);

      FailedFiles := '';

      { Embedding model }
      if not DownloadFile(
        'https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/onnx/model.onnx',
        EmbDir + '\model.onnx') then
        FailedFiles := FailedFiles + '  - Embedding model.onnx' + #13#10;

      { Embedding vocab }
      if not DownloadFile(
        'https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/vocab.txt',
        EmbDir + '\vocab.txt') then
        FailedFiles := FailedFiles + '  - Embedding vocab.txt' + #13#10;

      { Reranker model }
      if not DownloadFile(
        'https://huggingface.co/BAAI/bge-reranker-base/resolve/main/onnx/model.onnx',
        RerDir + '\model.onnx') then
        FailedFiles := FailedFiles + '  - Reranker model.onnx' + #13#10;

      { Reranker vocab }
      if not DownloadFile(
        'https://huggingface.co/BAAI/bge-reranker-base/resolve/main/vocab.txt',
        RerDir + '\vocab.txt') then
        FailedFiles := FailedFiles + '  - Reranker vocab.txt' + #13#10;

      if FailedFiles = '' then
      begin
        MsgBox('RAG 模型全部下载完成！' + #13#10 +
               '  Embedding: ' + EmbDir + #13#10 +
               '  Reranker:  ' + RerDir, mbInformation, MB_OK);
      end
      else
      begin
        MsgBox('部分 RAG 模型下载失败：' + #13#10 + FailedFiles + #13#10 +
               '请稍后手动运行 download_model.bat 重试。', mbError, MB_OK);
      end;
    end;
  end;
end;

[UninstallDelete]
Type: dirifempty; Name: "{app}\models\bge-small-zh"
Type: dirifempty; Name: "{app}\models\bge-reranker-base"
Type: dirifempty; Name: "{app}\models"
Type: dirifempty; Name: "{app}\data"
Type: dirifempty; Name: "{app}\icons"
