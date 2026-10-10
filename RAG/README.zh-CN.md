# PDF RAG Assistant

[English](README.md) | 中文

向你的 PDF 文档提问，得到注明出处（具体到页码）的回答。这是一个从零实现的检索增强生成（RAG）系统，没有用 LangChain 或 LlamaIndex，每一步都是清晰可读的 Python 代码，包括 PDF 解析、文本切分、embedding、FAISS / Milvus 向量检索、BM25 混合检索、HyDE、问题分解，以及带引用的答案生成。项目附带 Streamlit 聊天界面和命令行工具。

## 功能

- **PDF 解析**：基于 PyMuPDF，按阅读顺序提取文字，把排版折行合并回完整段落，中英文换行分别处理
- **递归切分**：优先在段落、句子、分句处断开，相邻块之间有重叠
- **8 家 embedding 服务**：OpenAI、通义（阿里云百炼）、智谱、硅基流动、Gemini、Voyage AI、Ollama，或用 sentence-transformers 在本地运行模型
- **两种向量库，接口统一**：**FAISS**（本地索引文件）或 **Milvus**（Milvus Lite、Milvus 服务或 Zilliz Cloud）
- **可自由组合的检索技术**：
  - **混合检索**：BM25 关键词检索加向量检索，用倒数排名融合（RRF）合并结果。既能匹配意思相近的说法，也能找到人名、编号、错误码这类原词
  - **HyDE**（假设文档嵌入）：先让大模型写一段"假设的答案"，再用它去检索
  - **问题分解**：把复杂问题拆成几个子问题，分别检索
  - **相似度阈值**：相关性太低的段落不会交给大模型
- **带引用的回答**：大模型只根据检索到的段落作答，用 [1]、[2] 标注出处；文档里没有的内容会明确说明；回答流式输出
- **7 家对话模型**：OpenAI、Claude、通义、DeepSeek、智谱、Gemini、Ollama

## 工作流程

```
建索引   PDF ─► 页 ─► 文本块 ─► 向量 ─► FAISS / Milvus
                              └──────► BM25 索引（基于同一批文本块）

回答     问题 ─► [问题分解] ─► 对每个（子）问题：
                    向量检索（可选 HyDE）─► 相似度阈值 ─┐
                    BM25 关键词检索 ────────────────────┴─► RRF 融合
                 ─► 取前 k 段 ─► 大模型 ─► 带 [1] [2] 引用的回答
```

## 快速开始

需要 Python 3.10 及以上版本。

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
pip install -r requirements.txt
cp .env.example .env          # 然后去掉注释，填上你要用的 API key
```

把 PDF 放进 `documents/` 文件夹，然后启动界面：

```bash
streamlit run app.py
```

在侧边栏点 **Index new files** 建索引，然后在聊天框提问。默认 embedding 和回答都用 OpenAI，可以在 **Models** 里换成其他服务。

### 命令行

```bash
python -m rag ingest                       # 索引 documents/ 里的新 PDF（--rebuild 全部重建）
python -m rag ask "报告的结论是什么？"
python -m rag search "错误码 E1234"         # 只显示检索到的段落，不生成回答
```

所有命令都支持和界面相同的选项，例如：

```bash
python -m rag ingest --embedding-provider qwen --store milvus
python -m rag ask "..." --embedding-provider qwen --store milvus --llm-provider claude --hyde --decompose
```

完整选项见 `python -m rag <命令> --help`。

## 配置

### 模型服务

在 `.env` 里填上你要用的服务的 API key（参考 `.env.example`）。

| provider | Embedding（默认模型） | 对话（默认模型） | API key 环境变量 |
|---|---|---|---|
| `openai` | text-embedding-3-small | gpt-6-luna | `OPENAI_API_KEY` |
| `claude` | （无 embedding 接口） | claude-opus-5-5 | `ANTHROPIC_API_KEY` |
| `qwen` | text-embedding-v4 | qwen-plus | `DASHSCOPE_API_KEY` |
| `deepseek` | （无 embedding 接口） | deepseek-v4-flash | `DEEPSEEK_API_KEY` |
| `zhipu` | embedding-3 | glm-5.3 | `ZHIPUAI_API_KEY` |
| `siliconflow` | BAAI/bge-m3 | | `SILICONFLOW_API_KEY` |
| `gemini` | gemini-embedding-001 | gemini-3.6-flash | `GEMINI_API_KEY` |
| `voyage` | voyage-4 | | `VOYAGE_API_KEY` |
| `ollama` | bge-m3（本地） | qwen3（本地） | 不需要 |
| `local` | BAAI/bge-m3（sentence-transformers） | | 不需要 |

想用某家的其他模型，填上模型名就行（命令行的 `--embedding-model`、`--llm-model`，或界面里的输入框）。其他兼容 OpenAI 接口的服务，在 `rag/embedding.py` / `rag/llm.py` 的服务表里加一行即可接入。

> **换 embedding 模型需要重建索引。** 不同模型生成的向量不能混用，所以向量库会记录建库时用的模型，发现不一致会直接报错。点 **Rebuild index**，或运行 `python -m rag ingest --rebuild`。

### 向量库

| | FAISS（默认） | Milvus |
|---|---|---|
| 准备工作 | 无 | 用 Milvus Lite 无需准备；也可以部署 [Milvus 服务](https://milvus.io/docs/install_standalone-docker.md) |
| 数据存放位置 | `index/` | `index/milvus.db`（Lite）或服务端 |
| 怎么选用 | `--store faiss` | `--store milvus --milvus-uri <地址>` |

Milvus 的部署方式由地址决定：
- 填 `index/milvus.db` 这样的文件路径，就在进程内运行 Milvus Lite。
- 填 `http://localhost:19530`，就连接 Milvus 服务。
- 填 Zilliz Cloud 的地址，同时在 `.env` 里配上 `MILVUS_TOKEN`，就用托管服务。

Milvus Lite 的数据库文件同一时间只能被一个进程打开，所以界面开着的时候，不要再用命令行操作同一个文件。

### 检索选项

| 选项 | 界面控件 | 命令行参数 | 默认值 |
|---|---|---|---|
| 交给大模型的段落数 | Passages per answer | `-k` | 4 |
| 相似度阈值 | Similarity threshold | `--threshold` | 0.5 |
| BM25 + 向量混合检索 | Hybrid search | `--no-hybrid` 关闭 | 开 |
| 混合检索中向量结果的权重 | Embedding weight | `--dense-weight` | 0.5 |
| HyDE | HyDE | `--hyde` | 关 |
| 问题分解 | Query decomposition | `--decompose` | 关 |

**关于阈值**：它是问题和段落之间余弦相似度的最低要求。默认设得偏严，避免弱相关的段落干扰大模型。但不同 embedding 模型的分数范围不一样，OpenAI 的 text-embedding-3 系列普遍比 BGE、通义的分数低。如果回答说"文档里没有"，而其实文档里有，就把阈值调低一些，用 OpenAI embedding 的话可以从 0.3 开始试。开启混合检索时，阈值只过滤向量检索的结果，BM25 会补充关键词匹配；但如果没有任何段落达到阈值，就直接认为问题和文档无关。

HyDE 和问题分解每次提问都会多调用几次大模型，回答会更慢、成本更高；混合检索不额外调用任何 API。

## 项目结构

```
├── app.py                 Streamlit 聊天界面
├── documents/             放你的 PDF（不会提交到 git）
├── index/                 FAISS 索引 / Milvus Lite 数据库（首次索引时自动创建，不提交）
├── rag/
│   ├── __main__.py        命令行：python -m rag ingest | search | ask
│   ├── chain.py           RAG 链、整体配置、建索引
│   ├── pdf_loader.py      PDF 解析
│   ├── chunker.py         递归文本切分
│   ├── embedding.py       各家 embedding 服务
│   ├── llm.py             各家对话模型、流式输出
│   ├── vector_store.py    向量库接口 + FAISS 实现
│   ├── milvus_store.py    Milvus 实现
│   ├── bm25.py            BM25 关键词索引
│   ├── retriever.py       混合检索、HyDE、问题分解、阈值
│   └── document.py        各步骤之间传递的 Document 类型
├── requirements.txt
└── .env.example           API key 模板
```

部分模块可以单独运行用来调试，例如 `python -m rag.pdf_loader documents/` 预览提取出的文字，`python -m rag.chunker documents/` 预览切分结果。

## 局限

- 扫描版 PDF（文字是图片）提取不到文字，会被跳过并给出警告，需要先做 OCR。
- 每页单独切分，跨页的段落会被分到两个块里。
- 表格会被当作普通文字提取，行列结构会丢失。
- 每个问题单独回答，追问（比如"那它的缺点呢？"）看不到上一个问题。

## 安装注意事项

- **macOS 13 及更早版本**：新版 `faiss-cpu` 没有这些系统的预编译包，pip 会尝试从源码编译然后失败。请用 `pip install --only-binary :all: faiss-cpu` 安装。
- **NumPy 1.x 环境**（比如较老的 Anaconda）：新版 `pyarrow`（Milvus Lite 和 Streamlit 都依赖它）要求 NumPy 2。可以升级 NumPy，或者用 `pip install "pyarrow<26"` 限定版本。
- **本地 embedding 模型**（`local`）需要 `pip install sentence-transformers`，会一并安装 PyTorch。
