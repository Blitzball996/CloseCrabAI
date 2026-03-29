; CloseCrab 安装脚本
#define MyAppName "CloseCrab"
#define MyAppVersion "1.0.0"
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
SetupIconFile=icons\closecrab.ico

[Files]
Source: "out\build\x64-Release\closecrab.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "out\build\x64-Release\*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "config\config.yaml"; DestDir: "{app}\config"; Flags: ignoreversion
Source: "download_model.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "run.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "icons\closecrab.ico"; DestDir: "{app}\icons"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\run.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0
Name: "{group}\模型下载器"; Filename: "{app}\download_model.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\run.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Run]
Filename: "{app}\download_model.bat"; Description: "下载 AI 模型 (推荐)"; Flags: postinstall runhidden; Check: ShouldDownloadModel

[Code]
var
  ModelPage: TWizardPage;
  ModelCombo: TNewComboBox;
  RAGCheckBox: TNewCheckBox;

function ShouldDownloadModel: Boolean;
begin
  Result := True;
end;

procedure InitializeWizard;
var
  Label1, Label2, RAGLabel, RAGDesc: TNewStaticText;
begin
  ModelPage := CreateCustomPage(wpSelectTasks, '选择 AI 模型', '请选择要下载的模型');
  
  Label1 := TNewStaticText.Create(ModelPage);
  Label1.Parent := ModelPage.Surface;
  Label1.Caption := 'CloseCrab 需要下载模型文件才能运行。';
  Label1.AutoSize := True;
  Label1.Top := 0;
  
  Label2 := TNewStaticText.Create(ModelPage);
  Label2.Parent := ModelPage.Surface;
  Label2.Caption := 'LLM 大语言模型：';
  Label2.AutoSize := True;
  Label2.Top := 25;
  Label2.Font.Style := [fsBold];
  
  ModelCombo := TNewComboBox.Create(ModelPage);
  ModelCombo.Parent := ModelPage.Surface;
  ModelCombo.Top := 45;
  ModelCombo.Width := 380;
  ModelCombo.Style := csDropDownList;
  ModelCombo.Items.Add('Qwen2.5-7B (推荐, 4.5GB, 8GB显存)');
  ModelCombo.Items.Add('Qwen2.5-14B (更强, 8.5GB, 12GB显存)');
  ModelCombo.Items.Add('Qwen2.5-3B (轻量, 2GB, 6GB显存)');
  ModelCombo.Items.Add('Qwen2.5-1.5B (极速, 1.2GB, 4GB显存)');
  ModelCombo.ItemIndex := 0;

  RAGLabel := TNewStaticText.Create(ModelPage);
  RAGLabel.Parent := ModelPage.Surface;
  RAGLabel.Caption := 'RAG 知识库检索模型：';
  RAGLabel.AutoSize := True;
  RAGLabel.Top := 90;
  RAGLabel.Font.Style := [fsBold];

  RAGCheckBox := TNewCheckBox.Create(ModelPage);
  RAGCheckBox.Parent := ModelPage.Surface;
  RAGCheckBox.Top := 110;
  RAGCheckBox.Width := 420;
  RAGCheckBox.Caption := '下载 RAG 模型 (Embedding ~96MB + Reranker ~1.1GB)';
  RAGCheckBox.Checked := True;

  RAGDesc := TNewStaticText.Create(ModelPage);
  RAGDesc.Parent := ModelPage.Surface;
  RAGDesc.Caption := '来自 onnx-community 预转换仓库' + #13#10 +
                     '每个模型包含: model.onnx + model.onnx_data + tokenizer.json' + #13#10 +
                     '提示：模型下载需要几分钟到几小时，取决于网速。';
  RAGDesc.AutoSize := True;
  RAGDesc.Top := 135;
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

function DownloadFile(Url, DestPath: string): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('curl', '-L -o "' + DestPath + '" "' + Url + '"',
                 '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := Result and (ResultCode = 0);
end;

function UpdateConfigFile(ConfigPath, ModelName: string): Boolean;
var
  Lines: TStringList;
begin
  Result := False;
  Lines := TStringList.Create;
  try
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
                  '  embedding_model_path: "models/bge-small-zh/onnx/model.onnx"'#13#10 +
                  '  embedding_tokenizer_path: "models/bge-small-zh/tokenizer.json"'#13#10 +
                  '  reranker_model_path: "models/bge-reranker-base/onnx/model.onnx"'#13#10 +
                  '  reranker_tokenizer_path: "models/bge-reranker-base/tokenizer.json"';
    Lines.SaveToFile(ConfigPath);
    Result := True;
  finally
    Lines.Free;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ModelUrl, ModelName, ConfigPath, ModelDir: string;
  EmbOnnxDir, RerOnnxDir: string;
  FailedFiles: string;
begin
  if CurStep = ssPostInstall then
  begin
    ModelUrl := GetSelectedModelUrl;
    ModelName := GetSelectedModelName;
    ConfigPath := ExpandConstant('{app}\config\config.yaml');
    ModelDir := ExpandConstant('{app}\models');
    CreateDir(ModelDir);

    UpdateConfigFile(ConfigPath, ModelName);

    { 下载 LLM }
    if not DownloadFile(ModelUrl, ModelDir + '\' + ModelName) then
      MsgBox('LLM 下载失败，请稍后运行 download_model.bat', mbError, MB_OK)
    else
      MsgBox('LLM 下载完成！', mbInformation, MB_OK);

    { 下载 RAG 模型 }
    if RAGCheckBox.Checked then
    begin
      EmbOnnxDir := ModelDir + '\bge-small-zh\onnx';
      RerOnnxDir := ModelDir + '\bge-reranker-base\onnx';
      CreateDir(ModelDir + '\bge-small-zh');
      CreateDir(EmbOnnxDir);
      CreateDir(ModelDir + '\bge-reranker-base');
      CreateDir(RerOnnxDir);

      FailedFiles := '';

      { Embedding: model.onnx }
      if not DownloadFile(
        'https://huggingface.co/onnx-community/bge-small-zh-v1.5-ONNX/resolve/main/onnx/model.onnx',
        EmbOnnxDir + '\model.onnx') then
        FailedFiles := FailedFiles + '  - Embedding model.onnx' + #13#10;

      { Embedding: model.onnx_data }
      DownloadFile(
        'https://huggingface.co/onnx-community/bge-small-zh-v1.5-ONNX/resolve/main/onnx/model.onnx_data',
        EmbOnnxDir + '\model.onnx_data');

      { Embedding: tokenizer.json }
      if not DownloadFile(
        'https://huggingface.co/onnx-community/bge-small-zh-v1.5-ONNX/resolve/main/tokenizer.json',
        ModelDir + '\bge-small-zh\tokenizer.json') then
        FailedFiles := FailedFiles + '  - Embedding tokenizer.json' + #13#10;

      { Reranker: model.onnx }
      if not DownloadFile(
        'https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/onnx/model.onnx',
        RerOnnxDir + '\model.onnx') then
        FailedFiles := FailedFiles + '  - Reranker model.onnx' + #13#10;

      { Reranker: model.onnx_data }
      DownloadFile(
        'https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/onnx/model.onnx_data',
        RerOnnxDir + '\model.onnx_data');

      { Reranker: tokenizer.json }
      if not DownloadFile(
        'https://huggingface.co/onnx-community/bge-reranker-base-ONNX/resolve/main/tokenizer.json',
        ModelDir + '\bge-reranker-base\tokenizer.json') then
        FailedFiles := FailedFiles + '  - Reranker tokenizer.json' + #13#10;

      if FailedFiles = '' then
        MsgBox('RAG 模型全部下载完成！', mbInformation, MB_OK)
      else
        MsgBox('部分 RAG 文件下载失败：' + #13#10 + FailedFiles + '请稍后运行 download_model.bat', mbError, MB_OK);
    end;
  end;
end;

[UninstallDelete]
Type: dirifempty; Name: "{app}\models\bge-small-zh\onnx"
Type: dirifempty; Name: "{app}\models\bge-small-zh"
Type: dirifempty; Name: "{app}\models\bge-reranker-base\onnx"
Type: dirifempty; Name: "{app}\models\bge-reranker-base"
Type: dirifempty; Name: "{app}\models"
Type: dirifempty; Name: "{app}\data"
Type: dirifempty; Name: "{app}\icons"
