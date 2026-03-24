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

; ===== 添加图标配置 =====
; 安装程序本身的图标（显示在 .exe 安装文件上）
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

; ===== 添加图标文件到安装包 =====
; 将图标文件复制到安装目录，供快捷方式使用
Source: "icons\closecrab.ico"; DestDir: "{app}\icons"; Flags: ignoreversion

[Icons]
; 开始菜单快捷方式 - 使用独立的图标文件
Name: "{group}\{#MyAppName}"; Filename: "{app}\run.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0
Name: "{group}\模型下载器"; Filename: "{app}\download_model.bat"; IconFilename: "{app}\icons\closecrab.ico"; IconIndex: 0
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; 桌面快捷方式 - 使用独立的图标文件
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

function ShouldDownloadModel: Boolean;
begin
  Result := True;
end;

procedure InitializeWizard;
var
  Label1, Label2: TNewStaticText;
begin
  { 创建自定义页面 }
  ModelPage := CreateCustomPage(wpSelectTasks, '选择 AI 模型', '请选择要下载的模型');
  
  Label1 := TNewStaticText.Create(ModelPage);
  Label1.Parent := ModelPage.Surface;
  Label1.Caption := 'CloseCrab 需要 GGUF 格式的模型文件才能运行。';
  Label1.AutoSize := True;
  Label1.Top := 0;
  
  Label2 := TNewStaticText.Create(ModelPage);
  Label2.Parent := ModelPage.Surface;
  Label2.Caption := '请选择要下载的模型：' + #13#10 + 
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
  
  ModelDescLabel := TNewStaticText.Create(ModelPage);
  ModelDescLabel.Parent := ModelPage.Surface;
  ModelDescLabel.Caption := '提示：模型下载需要几分钟到几小时，取决于网速。';
  ModelDescLabel.AutoSize := True;
  ModelDescLabel.Top := 150;
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
  Modified: Boolean;
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
                    '  temperature: 0.7';
    
    Modified := False;
    for i := 0 to Lines.Count - 1 do
    begin
      if Pos('model_path:', Lines[i]) > 0 then
      begin
        Lines[i] := '  model_path: "models/' + ModelName + '"';
        Modified := True;
      end;
    end;
    
    if not Modified then
    begin
      Lines.Add('llm:');
      Lines.Add('  model_path: "models/' + ModelName + '"');
      Lines.Add('  max_tokens: 512');
      Lines.Add('  temperature: 0.7');
    end;
    
    Lines.SaveToFile(ConfigPath);
    Result := True;
  finally
    Lines.Free;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ModelUrl, ModelName, ConfigPath: string;
  ModelDir: string;
begin
  if CurStep = ssPostInstall then
  begin
    ModelUrl := GetSelectedModelUrl;
    ModelName := GetSelectedModelName;
    ConfigPath := ExpandConstant('{app}\config\config.yaml');
    
    ModelDir := ExpandConstant('{app}\models');
    CreateDir(ModelDir);
    
    UpdateConfigFile(ConfigPath, ModelName);
    
    if not Exec('curl', '-L -o "' + ModelDir + '\' + ModelName + '" "' + ModelUrl + '"',
                '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    begin
      MsgBox('无法启动下载工具，请手动下载模型。' + #13#10 +
             '下载地址: ' + ModelUrl, mbError, MB_OK);
    end
    else if ResultCode = 0 then
    begin
      MsgBox('模型下载完成！' + #13#10 + '文件保存在: ' + ModelDir + '\' + ModelName,
             mbInformation, MB_OK);
    end
    else
    begin
      MsgBox('模型下载失败，请检查网络后手动下载。' + #13#10 +
             '下载地址: ' + ModelUrl, mbError, MB_OK);
    end;
  end;
end;

[UninstallDelete]
Type: dirifempty; Name: "{app}\models"
Type: dirifempty; Name: "{app}\data"
Type: dirifempty; Name: "{app}\icons"