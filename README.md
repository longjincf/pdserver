# 步健慧测（PD Server）

本项目是华南师范大学 2025 年省国级大创项目立项《基于多模态学习的帕金森病症分析评估系统》的 web 端软件，项目编号 S202510574149。

本软件具备自主知识产权，软件著作权登记号为2026SR0435871。

本项目遵循 GPLv3 开源协议，可以自由使用、修改、学习代码；如果基于此项目做衍生品并分发，必须按照协议用 GPLv3 开源。

**团队成员：**

| 角色                 | 姓名   | 学号        |
| -------------------- | ------ | ----------- |
| 指导老师（讲师）     | 周成菊 |             |
| 指导老师（主治医师） | 黄卓群 |             |
| 负责人               | 龙成飞 | 20232034061 |
| 成员                 | 韦彦修 | 20232034045 |
| 成员                 | 赵捷   | 20232005051 |
| 成员                 | 林伟都 | 20232034051 |

---

本项目是基于 Django + REST Framework 的帕金森病多模态评估系统，通过**语音、面部、步态**三种方式进行辅助评估，返回 0–4 级严重程度分级与症状描述。

系统支持 **LSTM** 和 **Random Forest** 双模型引擎，LSTM 优先加载、RF 自动降级，API 接口完全透明。

---

## 目录

- [功能概览](#功能概览)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速部署](#快速部署)
- [API 接口](#api-接口)
- [视频处理与模型实现](#视频处理与模型实现)
- [模型训练](#模型训练)
- [生产环境部署](#生产环境部署)
- [开源协议](#开源协议)

---

## 功能概览

| 评估方式 | 输入     | 提取特征                       | 模型            | 输出                 |
| -------- | -------- | ------------------------------ | --------------- | -------------------- |
| 语音评估 | WAV 音频 | 逐帧 Jitter / Shimmer / HNR    | LSTM / RF       | UPDRS + 0–4 分级    |
| 面部评估 | 视频文件 | 逐帧 EAR / 表情概率 / 嘴唇距离 | LSTM / 规则打分 | 0–4 分级 + 症状描述 |
| 步态评估 | 视频文件 | 逐帧踝关节 / 髋部 5 维轨迹     | LSTM / RF       | UPDRS + 0–4 分级    |

系统同时提供 Web 页面（桌面端/移动端）和 REST API，可对接微信小程序等前端应用。

---

## 技术栈

- **后端框架**：Django 4.2 + Django REST Framework
- **语音分析**：Praat-Parselmouth（声学特征提取）
- **面部分析**：InsightFace（人脸检测与 106 点关键点）+ ONNX Runtime（表情识别）
- **步态分析**：MediaPipe Pose（人体姿态估计）
- **预测模型**：PyTorch LSTM（时序回归/分类）+ scikit-learn RandomForest（兼容降级）
- **视频处理**：OpenCV（逐帧采样与图像处理）

---

## 项目结构

```
pdserver/
├── api/                              # 主应用
│   ├── views.py                      # API 视图（LSTM 优先 / RF 自动降级）
│   ├── urls.py                       # API 路由
│   ├── models.py                     # 数据库模型（AssessmentResult）
│   ├── feature_extractor.py          # 语音特征提取（全局 + 逐帧时序）
│   ├── feature_extractor_face.py     # 面部视频分析（聚合 + 逐帧时序）
│   ├── feature_extractor_gait.py     # 步态视频分析（聚合 + 逐帧时序）
│   ├── modeling.py                   # 语音预测（LSTM / RF 双模型切换）
│   ├── gait_modeling.py              # 步态预测（LSTM / RF 双模型切换）
│   ├── face_modeling.py              # 面部预测（LSTM / 规则打分切换）
│   ├── symptoms.py                   # 严重程度分级与症状描述
│   ├── lstm_models/                  # LSTM 模型模块
│   │   ├── base.py                   # 训练基础设施（训练循环、早停、padding、保存加载）
│   │   ├── gait_lstm.py              # 步态 LSTM 回归模型（5 维输入 → UPDRS）
│   │   ├── face_lstm.py              # 面部 LSTM 分类模型（9 维输入 → 0-4 级）
│   │   └── voice_lstm.py             # 语音 LSTM 回归模型（3 维输入 → UPDRS）
│   ├── models/                       # 预置的推理模型文件
│   │   ├── buffalo_l/                # InsightFace 人脸检测模型（ONNX）
│   │   ├── mini_xception.onnx        # 表情识别模型
│   │   └── emotion-ferplus-8.onnx
│   └── management/commands/
│       ├── train_updrs.py            # 语音模型训练（--model rf / lstm）
│       ├── train_gait.py             # 步态模型训练（--model rf / lstm）
│       └── train_face.py             # 面部模型训练（--model lstm）
├── pdserver/                         # Django 项目配置
│   ├── settings.py                   # 全局配置
│   ├── urls.py                       # 根路由
│   ├── wsgi.py / asgi.py             # 部署入口
├── model/                            # 模型文件目录
│   ├── model.pkl                     # 语音 RF 模型
│   ├── voice_lstm.pt                 # 语音 LSTM 模型
│   ├── gait_model.pkl                # 步态 RF 模型
│   ├── gait_lstm.pt                  # 步态 LSTM 模型
│   └── face_lstm.pt                  # 面部 LSTM 模型
├── templates/                        # 前端页面
│   ├── home.html                     # 首页
│   ├── voice.html / face.html / gait.html
├── static/                           # 静态资源
├── manage.py                         # Django 管理脚本
├── requirements.txt                  # Python 依赖
└── db.sqlite3                        # SQLite 数据库（开发默认）
```

---

## 快速部署

### 1. 环境准备

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 初始化数据库

```bash
python manage.py migrate
```

### 4. 启动服务

```bash
python manage.py runserver 0.0.0.0:8000
```

启动后访问：

| 页面     | 地址                        |
| -------- | --------------------------- |
| 首页     | http://localhost:8000/      |
| 语音评估 | http://localhost:8000/voice |
| 面部评估 | http://localhost:8000/face  |
| 步态评估 | http://localhost:8000/gait  |

RF 模型文件（`.pkl`）和 ONNX 推理模型已随项目预置，**无需训练即可直接使用**。训练 LSTM 模型后会自动优先加载。

---

## API 接口

所有接口均以 `/api/` 为前缀。每个预测接口的返回中会包含 `"model_type"` 字段，标识实际使用的模型（`lstm` / `rf` / `rule` / `heuristic`）。

### 评估接口

| 接口                  | 方法 | 参数                  | 说明                                |
| --------------------- | ---- | --------------------- | ----------------------------------- |
| `/api/predict`      | POST | `audio`（WAV/MP3）  | 语音评估：返回 UPDRS + 0–4 分级    |
| `/api/face_predict` | POST | `video`（视频文件） | 面部评估：返回 0–4 分级 + 症状描述 |
| `/api/gait_predict` | POST | `video`（视频文件） | 步态评估：返回 UPDRS + 0–4 分级    |

### 数据管理

| 接口                             | 方法 | 说明                          |
| -------------------------------- | ---- | ----------------------------- |
| `/api/health`                  | GET  | 健康检查                      |
| `/api/model_info`              | GET  | 各模型状态与类型              |
| `/api/save_data`               | POST | 手动保存评估记录              |
| `/api/get_all_data`            | GET  | 获取全部评估记录              |
| `/api/get_latest_data`         | GET  | 各类型最新一条记录            |
| `/api/get_data/<id>`           | GET  | 按 ID 查询单条                |
| `/api/get_data_by_type/<type>` | GET  | 按类型查询（voice/face/gait） |

---

## 视频处理与模型实现

### 视频处理流程

系统对上传的视频采用**逐帧采样**策略，同时产出两条数据通道：

**面部视频**（`feature_extractor_face.py`）：

```
视频 → 3 FPS 采样 → InsightFace 106 点关键点
     → 逐帧计算: EAR(眨眼) + 嘴唇距离 + 7 类表情概率
     → 通道 1: 聚合统计量（眨眼频率、表情熵、嘴唇张开比）→ 规则打分
     → 通道 2: 原始帧序列 (N, 9) → LSTM 分类
```

**步态视频**（`feature_extractor_gait.py`）：

```
视频 → 10 FPS 采样 → MediaPipe Pose 33 关键点
     → 逐帧追踪: 左右踝 X/Y + 髋部 Y
     → 通道 1: 聚合统计量（步数、步频、对称性等 20+ 特征）→ RF 回归
     → 通道 2: 归一化帧序列 (N, 5) → LSTM 回归
```

**语音音频**（`feature_extractor.py`）：

```
WAV → 全局分析: Jitter / Shimmer / HNR → RF 回归
    → 分帧分析: 25ms 窗口 / 10ms 步长 → 逐帧时序 (N, 3) → LSTM 回归
```

### 模型加载策略

```
请求到达
  ↓
同时提取聚合特征 + 原始时序
  ↓
加载模型: .pt (LSTM) 存在？→ 是 → 用 LSTM + 时序数据预测
                          → 否 → .pkl (RF) 存在？→ 是 → 用 RF + 聚合特征预测
                                                → 否 → 启发式规则 / 规则打分
```

所有 `load_model()` 函数遵循同一策略，对 API 调用方完全透明。

### LSTM 模型结构

| 模块      | 网络结构                         | 输入维度         | 输出               |
| --------- | -------------------------------- | ---------------- | ------------------ |
| 步态 LSTM | LSTM(128, 2层) → FC(128→64→1) | `(seq_len, 5)` | UPDRS 回归值       |
| 面部 LSTM | LSTM(64, 2层) → FC(64→32→5)   | `(seq_len, 9)` | 5 类 logits（0-4） |
| 语音 LSTM | LSTM(128, 2层) → FC(128→64→1) | `(seq_len, 3)` | UPDRS 回归值       |

三个模型均通过 `api/lstm_models/base.py` 中的 `BaseLSTMTrainer` 提供统一的训练循环（早停、学习率调度、变长序列 padding/masking、模型保存加载）。

### RF 模型（兼容保留）

语音和步态的 RF 模型采用 `StandardScaler + RandomForestRegressor(n_estimators=300)` Pipeline，作为 LSTM 不可用时的自动降级方案。

---

## 模型训练

### 训练命令

```bash
# ── 语音 ──
# RF（默认，使用 UCI telemonitoring CSV）
set PD_DATA_PATH=C:\path\to\parkinsons_updrs.data
python manage.py train_updrs --model rf

# LSTM（需要帧级时序 CSV 或兼容旧格式）
python manage.py train_updrs --model lstm --epochs 100 --batch-size 32

# ── 步态 ──
# RF
set GAIT_DATA_PATH=C:\path\to\gait_features.csv
python manage.py train_gait --model rf

# LSTM
python manage.py train_gait --model lstm --epochs 100

# ── 面部 ──
# 仅 LSTM，需要含 sequence + grade 列的 CSV
python manage.py train_face --data face_data.csv --epochs 100
```

### 训练数据格式

**RF 模型**：直接使用原有 CSV 格式（聚合统计量 + 标签列）。

**LSTM 模型**：推荐使用包含 `sequence` 列的 CSV，每行存储一个 JSON 格式的时序数组。如果没有 `sequence` 列，训练器会自动从聚合特征构造伪时序（兼容旧格式，但效果不如真实时序数据）。

### 模型文件

训练完成后模型自动保存到 `model/` 目录：

| 文件               | 模型类型  | 优先级               |
| ------------------ | --------- | -------------------- |
| `voice_lstm.pt`  | 语音 LSTM | 高（优先加载）       |
| `model.pkl`      | 语音 RF   | 低（降级使用）       |
| `gait_lstm.pt`   | 步态 LSTM | 高                   |
| `gait_model.pkl` | 步态 RF   | 低                   |
| `face_lstm.pt`   | 面部 LSTM | 高（降级到规则打分） |

---

## 生产环境部署

### 使用 Gunicorn

```bash
pip install gunicorn
gunicorn pdserver.wsgi:application --bind 0.0.0.0:8000 --workers 4
```

### 配合 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /path/to/pdserver/static/;
    }
}
```

### 数据库

开发环境默认 SQLite。生产环境可在 `pdserver/settings.py` 中切换为 MySQL 或 PostgreSQL。

### 注意事项

- 上传文件大小限制为 10MB
- CORS 已配置，支持微信小程序（`servicewechat.com`）调用
- 时区默认 `Asia/Shanghai`
- LSTM 推理支持 CPU，无需 GPU

---

## 开源协议

本项目采用 [GNU General Public License v3.0](LICENSE) 开源协议。

- 你可以自由使用、修改和分发本软件，但衍生作品必须同样以 GPLv3 协议开源
- 本项目使用了 GPLv3+ 协议的 [Praat-Parselmouth](https://github.com/YannickJadoul/Parselmouth) 库，因此整体作品需以 GPL 兼容协议发布
- 本项目已登记软件著作权（登记号 2026SR0435871），开源协议与著作权并行不悖

第三方依赖及其协议详见 [NOTICE](NOTICE) 文件。

特别说明：
- InsightFace 预训练模型（`api/models/buffalo_l/`）仅供非商业研究用途，商业使用需另行获取授权
- UCI Parkinson's 数据集采用 CC BY 4.0 协议，使用时请注明出处
