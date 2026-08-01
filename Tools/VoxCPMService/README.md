# VCBM VoxCPM2 Local Service

## ユニット（関数）構造図

## 1. 全体構成

```mermaid
flowchart TD
    Unity[Unityクライアント]
    Server[server.py]
    Routes[routes.py]
    Services[services.py]
    VoxCPM[VoxCPM2]
    Output[outputフォルダ]

    Unity -->|HTTPリクエスト| Routes
    Server -->|FastAPIアプリを読み込み| Routes
    Routes -->|処理を呼び出す| Services
    Services -->|モデル読込・音声生成| VoxCPM
    Services -->|WAV保存| Output
    Routes -->|WAV取得| Output
    Routes -->|HTTPレスポンス| Unity
```

---

## 2. ファイル別構造

```text
server.py
└─ メイン処理
   └─ uvicorn.run()
      └─ routes.py の app を起動

routes.py
├─ GenerateRequest
│
│  サーバー、モデル、CUDAの状態を返す。（GET /health）
├─ health()
│
│  VoxCPM2モデルの読み込み状態を返す。（GET /model/status）
├─ model_status()
│
│  VoxCPM2モデルを明示的に読み込む。（POST /model/load）
├─ load_model()
│
│  Voice Designによる音声生成を実行する。（POST /voice/generate）
├─ voice_generate()
│
│  outputフォルダ内のWAVを参照して、同じ声質の音声を生成する。（POST /voice/clone）
├─ voice_clone()
│
│  生成済みWAVファイルを返す。（GET /audio/{file_name}） 
└─ get_audio()

services.py
├─ GenerationBusyError
├─ set_seed()
├─ get_module_devices()
├─ build_voice_design_text()
├─ build_clone_text()
├─ get_model()
├─ get_service_status()
├─ generate_voice()
│  ├─ get_model()
│  ├─ build_voice_design_text()
│  ├─ set_seed()
│  ├─ VoxCPM.generate()
│  └─ soundfile.write()
│
└─ get_audio_path()
```

---

## 3. `server.py` の構造

```mermaid
flowchart TD
    Main[メイン処理]
    ImportApp[routes.pyからappを読み込む]
    Run[uvicorn.run]
    FastAPI[FastAPIサーバー起動]

    Main --> ImportApp
    ImportApp --> Run
    Run --> FastAPI
```

### ユニット一覧

| ユニット            | 処理内容                 |
| --------------- | -------------------- |
| メイン処理           | Pythonから直接実行されたか確認する |
| `uvicorn.run()` | FastAPIサーバーを起動する     |

### 呼び出し関係

```text
メイン処理
└─ uvicorn.run(
       app=routes.app,
       host="127.0.0.1",
       port=8765
   )
```

---

## 4. `routes.py` の構造

```mermaid
flowchart TD
    Request[UnityからのHTTPリクエスト]

    Health[health]
    Status[model_status]
    Load[load_model]
    Generate[voice_generate]
    Audio[get_audio]

    ServiceStatus[services.get_service_status]
    GetModel[services.get_model]
    GenerateVoice[services.generate_voice]
    GetAudioPath[services.get_audio_path]

    Request -->|GET /health| Health
    Request -->|GET /model/status| Status
    Request -->|POST /model/load| Load
    Request -->|POST /voice/generate| Generate
    Request -->|GET /audio/file_name| Audio

    Health --> ServiceStatus
    Status --> ServiceStatus

    Load --> GetModel
    Load --> ServiceStatus

    Generate --> GenerateVoice

    Audio --> GetAudioPath
```

### HTTP受付ユニット

| 関数                 | HTTPメソッド | URL                  | 呼び出す処理                 |
| ------------------ | -------- | -------------------- | ---------------------- |
| `health()`         | GET      | `/health`            | `get_service_status()` |
| `model_status()`   | GET      | `/model/status`      | `get_service_status()` |
| `load_model()`     | POST     | `/model/load`        | `get_model()`          |
| `voice_generate()` | POST     | `/voice/generate`    | `generate_voice()`     |
| `get_audio()`      | GET      | `/audio/{file_name}` | `get_audio_path()`     |

### `GenerateRequest`

音声生成APIで受け付けるデータを定義する。

```text
GenerateRequest
├─ voice_prompt
├─ speech_text
├─ cfg_value
├─ inference_timesteps
├─ seed
├─ candidate_count
└─ normalize
```

---

## 5. `services.py` の構造

```mermaid
flowchart TD
    GenerateVoice[generate_voice]

    Lock[生成ロック取得]
    GetModel[get_model]
    BuildText[build_voice_design_text]
    Loop[候補数分繰り返す]
    Seed[set_seed]
    Generate[VoxCPM.generate]
    Convert[NumPy配列へ変換]
    Validate[波形チェック]
    Save[soundfile.write]
    Result[結果情報を作成]
    Unlock[生成ロック解放]

    GenerateVoice --> Lock
    Lock --> GetModel
    GetModel --> BuildText
    BuildText --> Loop
    Loop --> Seed
    Seed --> Generate
    Generate --> Convert
    Convert --> Validate
    Validate --> Save
    Save --> Result
    Result -->|次の候補| Loop
    Result --> Unlock
```

---

## 6. モデル管理ユニット

```mermaid
flowchart TD
    Call[get_model呼び出し]
    Loaded{モデル読込済みか}
    ReturnExisting[既存モデルを返す]
    Lock[モデルロック取得]
    DoubleCheck{再確認}
    Loading[状態をloadingへ変更]
    Load[VoxCPM.from_pretrained]
    Devices[get_module_devices]
    Ready[状態をreadyへ変更]
    Return[モデルを返す]
    Error[状態をerrorへ変更]

    Call --> Loaded

    Loaded -->|はい| ReturnExisting
    Loaded -->|いいえ| Lock

    Lock --> DoubleCheck
    DoubleCheck -->|読込済み| ReturnExisting
    DoubleCheck -->|未読込| Loading

    Loading --> Load
    Load -->|成功| Devices
    Devices --> Ready
    Ready --> Return

    Load -->|失敗| Error
```

### `get_model()`

```text
get_model()
├─ モデルがすでに存在する
│  └─ 既存モデルを返す
│
└─ モデルが存在しない
   ├─ モデル読み込みロックを取得
   ├─ 状態を loading に設定
   ├─ VoxCPM.from_pretrained() を実行
   ├─ get_module_devices() で配置先を取得
   ├─ 状態を ready に設定
   └─ モデルを返す
```

---

## 7. 音声生成ユニット

### `generate_voice()`

```text
generate_voice()
├─ 生成ロック取得
│
├─ get_model()
│  └─ VoxCPM2モデルを取得
│
├─ build_voice_design_text()
│  └─ 音声プロンプトと発話文章を結合
│
├─ 候補数分繰り返す
│  ├─ set_seed()
│  ├─ model.generate()
│  ├─ NumPy配列へ変換
│  ├─ 空の波形でないか確認
│  ├─ ファイル名を作成
│  ├─ sf.write()
│  └─ 候補情報をリストへ追加
│
├─ 生成結果を返す
│
└─ 生成ロック解放
```

### 詳細な呼び出し関係

```mermaid
flowchart TD
    GenerateVoice[generate_voice]
    GetModel[get_model]
    BuildText[build_voice_design_text]
    SetSeed[set_seed]
    TorchSeed[torch.manual_seed]
    ModelGenerate[model.generate]
    Numpy[numpy.asarray]
    Write[soundfile.write]
    Candidate[候補情報作成]

    GenerateVoice --> GetModel
    GenerateVoice --> BuildText
    GenerateVoice --> SetSeed

    SetSeed --> TorchSeed

    GenerateVoice --> ModelGenerate
    ModelGenerate --> Numpy
    Numpy --> Write
    Write --> Candidate
```

---

## 8. 補助ユニット

### `set_seed()`

乱数シードを統一し、同じシードで近い生成結果を再現できるようにする。

```text
set_seed()
├─ random.seed()
├─ numpy.random.seed()
├─ torch.manual_seed()
└─ torch.cuda.manual_seed_all()
```

### `get_module_devices()`

モデルのパラメータとバッファが配置されているデバイスを取得する。

```text
get_module_devices()
├─ module.parameters()
├─ module.buffers()
└─ デバイス一覧を返す
```

戻り値の例：

```text
["cuda:0"]
```

または、

```text
["cpu"]
```

### `build_voice_design_text()`

音声設計プロンプトと発話文章を、VoxCPM2用の形式に変換する。

```text
入力
├─ voice_prompt
└─ speech_text

処理結果
└─ (voice_prompt)speech_text
```

例：

```text
voice_prompt:
A young Japanese girl with a bright and youthful voice.

speech_text:
今日はいい天気だね。
```

変換結果：

```text
(A young Japanese girl with a bright and youthful voice.)今日はいい天気だね。
```

### `get_service_status()`

モデルとGPUの状態を取得する。

```text
get_service_status()
├─ モデル状態
├─ モデルID
├─ 指定デバイス
├─ 実際のモデル配置先
├─ PyTorchバージョン
├─ CUDAバージョン
├─ CUDA利用可否
├─ GPU名
├─ GPUメモリ使用量
└─ エラー内容
```

### `get_audio_path()`

指定されたWAVファイルを安全に取得する。

```text
get_audio_path()
├─ ファイル名だけであることを確認
├─ outputフォルダ内のパスを作成
├─ ファイル存在確認
└─ Pathを返す
```

---

## 9. 音声生成時のシーケンス

```mermaid
sequenceDiagram
    participant Unity
    participant Routes as routes.py
    participant Services as services.py
    participant Model as VoxCPM2
    participant Output as output/

    Unity->>Routes: POST /voice/generate
    Routes->>Services: generate_voice()

    Services->>Services: 生成ロック取得
    Services->>Services: get_model()

    alt モデル未読込
        Services->>Model: from_pretrained()
        Model-->>Services: 読み込み済みモデル
    end

    Services->>Services: build_voice_design_text()

    loop candidate_count回
        Services->>Services: set_seed()
        Services->>Model: generate()
        Model-->>Services: waveform
        Services->>Output: WAVファイル保存
    end

    Services-->>Routes: 候補情報
    Routes-->>Unity: JSONレスポンス
```

---

## 10. 音声ファイル取得時のシーケンス

```mermaid
sequenceDiagram
    participant Unity
    participant Routes as routes.py
    participant Services as services.py
    participant Output as output/

    Unity->>Routes: GET /audio/{file_name}
    Routes->>Services: get_audio_path(file_name)
    Services->>Output: ファイル存在確認
    Output-->>Services: WAVファイル
    Services-->>Routes: ファイルパス
    Routes-->>Unity: FileResponse
```
