# 📚 个人知识库问答助手（RAG Knowledge Base）

基于 **RAG（检索增强生成）** 架构的本地个人知识库问答系统。将你的 PDF / Markdown 文档解析、切分、向量化后入库，提问时通过「关键词检索 + 语义向量检索」的混合召回找到最相关的资料片段，交给 **DeepSeek 大模型** 生成有依据、可溯源的回答。

## ✨ 功能特性

- **混合检索**：BM25 关键词检索（权重 0.3）+ BGE 向量语义检索（权重 0.7），通过 RRF（倒数排名融合）合并结果，兼顾精确术语匹配与语义泛化能力
- **本地向量库**：使用 Chroma 持久化存储，文档离网也能检索
- **中文 Embedding**：采用 BAAI `bge-small-zh-v1.5` 模型，针对中文语料优化，支持离线加载（模型已下载到 `models/` 目录）
- **智能解析缓存**：对文档集合计算 MD5 指纹（路径 + 大小 + 修改时间 + 切分参数），文档未变化时直接读取 `chunks_cache.pkl`，避免重复解析，秒级启动
- **图形化问答界面**：基于 Streamlit 的聊天界面，支持流式输出、参考来源展示、多轮对话记忆、清空会话、重新生成上一条回答
- **会话持久化**：聊天记录保存到本地 `chat_history.json`，重启应用后历史不丢失
- **两种使用方式**：Streamlit 网页版（`app.py`）和终端问答版（`query.py`）

## 🏗️ 项目结构

```
rag-knowledge-base/
├── app.py                 # Streamlit 网页问答界面（主入口）
├── query.py               # 终端命令行问答脚本（纯向量检索版）
├── hybrid_retriever.py    # 混合检索器核心模块（BM25 + 向量 + RRF 融合）
├── build_index.py         # 文档解析与索引构建模块（与检索器逻辑同源）
├── download_model.py      # 从 HF 镜像下载 bge-small-zh-v1.5 嵌入模型
├── test_all.py            # 依赖包导入自检脚本
├── requirements.txt       # Python 依赖清单（完整锁定版本）
├── docs/                  # 知识库文档目录（放入 PDF / Markdown）
│   └── 26《中学综合素质》重点笔记.pdf
├── models/                # 本地 Embedding 模型（bge-small-zh-v1.5）
├── chroma_db/             # Chroma 向量数据库持久化目录
├── chunks_cache.pkl       # 文档切分结果缓存（自动生成）
├── chat_history.json      # 聊天历史持久化文件（自动生成）
└── .env                   # 环境变量文件（需自行创建，见下文）
```

## 📄 各模块说明

| 文件 | 作用 |
|------|------|
| `app.py` | 主应用。拉起 Streamlit 界面，初始化 LLM 与混合检索链，处理问答、流式输出、来源展示、历史持久化与重新生成。侧边栏可配置 API Key、检索片段数 top_k、记忆轮数、单次提问最大字数 |
| `hybrid_retriever.py` | 核心检索模块。`load_chunks()` 负责文档加载、切分与指纹缓存；`build_hybrid_retriever()` 组装 BM25 检索器 + Chroma 向量检索器，用 `EnsembleRetriever` 做 0.3/0.7 加权融合 |
| `build_index.py` | 与 `hybrid_retriever.py` 逻辑基本相同的索引构建版本（区分在于嵌入模型从线上 `BAAI/bge-small-zh-v1.5` 加载，而 hybrid_retriever 从本地 `./models/` 加载） |
| `query.py` | 简易终端问答。直接加载 Chroma 向量库（纯向量检索，k=3），构建 `RetrievalQA` 链，在命令行循环问答，输入 `q` 退出 |
| `download_model.py` | 通过 hf-mirror 镜像把中文嵌入模型下载到 `./models/bge-small-zh-v1.5`，供离线使用 |
| `test_all.py` | 环境自检脚本，逐一 import 关键依赖并打印结果，用于确认虚拟环境安装是否完整 |

## 🚀 快速开始

### 1. 创建虚拟环境

```powershell
# 在项目根目录下执行
# 进入你克隆下来的项目目录
cd E:\AI_Personal_knowledge_base\rag-knowledge-base#这是我自己的主机目录，仅作参考

.\rag_env\Scripts\Activate.ps1

# 激活虚拟环境（Windows PowerShell）
.\rag_env\Scripts\Activate.ps1
# 若激活报错"执行策略"，可改用 cmd：
# .venv\Scripts\activate.bat
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

验证环境是否完整：

```powershell
python test_all.py
```

全部打印 `OK` 且最后显示 `所有包导入成功 ✅` 即安装成功。

### 3. 配置 DeepSeek API Key

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-你的密钥
```

复制 `.env.example` 为 `.env`，把 `your_deepseek_api_key_here` 替换成你自己的 DeepSeek API Key。

（也可运行时直接在网页侧边栏输入 API Key，无需 `.env`。）

API Key 可在 DeepSeek 开放平台申请：https://platform.deepseek.com/

### 4. 准备知识库文档

把需要问答的 **PDF** 或 **Markdown** 文件放入 `docs/` 目录（支持子目录）。

### 5. 启动应用

**方式一：网页版（推荐，混合检索）**

```powershell
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，在对话框提问即可。

**方式二：终端版（纯向量检索）**

```powershell
python query.py
```

**首次运行说明**：

- 首次会解析 `docs/` 下所有文档并生成 `chunks_cache.pkl` 与向量库，耗时取决于文档数量；之后启动命中缓存则几乎瞬时完成。
- 首次运行前，请执行 `python download_model.py` 下载 Embedding 模型到 `models/` 目录。
- 项目附带 `docs/sample.md` 示例文档，可自行替换为自己的 PDF / Markdown 资料。
- 嵌入模型已内置在 `models/` 目录，无需联网下载；若需在其他机器重新下载，运行 `python download_model.py`。
- 运行成功如下
  ![image-20260927231947037](C:\Users\hasee\AppData\Roaming\Typora\typora-user-images\image-20260927231947037.png)

### 6. 更新知识库

往 `docs/` 添加/删除/修改文件后，重启应用即可——指纹检测会自动发现变化并重新解析切分。若向量库中需要重建 embedding，可删除 `chroma_db/` 目录后重启。

## ⚙️ 关键技术细节

| 项 | 取值 / 说明 |
|----|------------|
| LLM | `deepseek-chat`（temperature=0.2，流式输出） |
| Embedding 模型 | `BAAI/bge-small-zh-v1.5`（本地离线加载） |
| 向量库 | Chroma（持久化于 `./chroma_db`） |
| 文本切分 | `RecursiveCharacterTextSplitter`，chunk_size=500，overlap=50，按中文标点优先分割 |
| 混合检索权重 | BM25 : 向量 = 0.3 : 0.7（EnsembleRetriever，RRF 融合） |
| 默认检索数 | top_k=5（网页端可在侧边栏 1~10 调节） |
| 对话记忆 | 侧边栏可调 1~10 轮，拼接进 prompt 的 `chat_history` |
| HF 镜像 | `HF_ENDPOINT=https://hf-mirror.com`（国内网络下载模型用） |

## ⚠️ 常见问题

- **提示 API Key 无效 / 401**：检查 `.env` 中 `DEEPSEEK_API_KEY` 或侧边栏输入是否正确。
- **402 / Insufficient Balance**：DeepSeek 账户余额不足，需充值。
- **模型下载失败**：确认网络可访问 hf-mirror.com，或直接使用项目自带的 `models/` 目录。
- **文档更新后回答没变化**：确认新文件是 `.pdf` 或 `.md` 后缀并位于 `docs/` 下；必要时删除 `chunks_cache.pkl` 和 `chroma_db/` 强制全量重建。
- **中文 Embedding 首次加载慢**：属正常现象（torch 模型初始化），后续经 `st.cache_resource` 缓存后不会重复加载。

## 📦 依赖概览

完整锁定版本见 `requirements.txt`，核心依赖：

- `langchain` / `langchain-community` / `langchain-classic` / `langchain-deepseek` / `langchain-text-splitters` — RAG 编排框架
- `chromadb` — 本地向量数据库
- `sentence-transformers` / `torch` / `transformers` — 中文 Embedding 推理
- `rank-bm25` — BM25 关键词检索
- `pypdf` — PDF 解析
- `streamlit` — Web 问答界面
- `python-dotenv` — `.env` 环境变量管理
