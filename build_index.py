import os
import pickle
import hashlib
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

CHUNKS_CACHE = "./chunks_cache.pkl"


# ==================================================
# 计算指纹：文件信息 + 切分参数
# ==================================================
def compute_fingerprint(docs_dir, chunk_size, chunk_overlap):
    hasher = hashlib.md5()

    # 1. 文件信息（路径 + 大小 + 修改时间）
    for root, _, files in os.walk(docs_dir):
        for name in sorted(files):
            if not (name.lower().endswith(".pdf") or name.lower().endswith(".md")):
                continue
            path = os.path.join(root, name)
            stat = os.stat(path)
            info = f"{path}|{stat.st_size}|{stat.st_mtime}"
            hasher.update(info.encode("utf-8"))

    # 2. 切分参数
    hasher.update(f"chunk_size={chunk_size}|chunk_overlap={chunk_overlap}".encode("utf-8"))

    return hasher.hexdigest()


# ==================================================
# 带失效判断的 chunks 加载
# ==================================================
def load_chunks(docs_dir="./docs", chunk_size=500, chunk_overlap=50):
    """加载并切分文档。如果缓存有效则直接读缓存，否则重新解析。"""

    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    current_fp = compute_fingerprint(docs_dir, chunk_size, chunk_overlap)

    # 尝试读缓存
    if os.path.exists(CHUNKS_CACHE):
        try:
            with open(CHUNKS_CACHE, "rb") as f:
                cached = pickle.load(f)
            if cached.get("fingerprint") == current_fp:
                print(f"✅ 缓存命中，直接加载 {len(cached['chunks'])} 个 chunk")
                return cached["chunks"]
            else:
                print("⚠️ 检测到文档或参数变化，缓存失效，重新解析...")
        except Exception as e:
            print(f"⚠️ 缓存读取失败（{e}），重新解析...")

    # 重新解析
    print("📄 正在解析文档并切分...")
    all_docs = []

    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    all_docs.extend(pdf_loader.load())

    md_loader = DirectoryLoader(
        docs_dir, glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    all_docs.extend(md_loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = [c for c in splitter.split_documents(all_docs) if c.page_content.strip()]

    # 写入缓存：把 chunks 和 fingerprint 一起存
    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump({"fingerprint": current_fp, "chunks": chunks}, f)

    print(f"💾 已解析并缓存 {len(chunks)} 个 chunk")
    return chunks


# ==================================================
# 构建混合检索器（复用缓存）
# ==================================================
def build_hybrid_retriever(docs_dir="./docs", chroma_dir="./chroma_db", top_k=5):
    """构建混合检索器：BM25 关键词检索 + 向量语义检索，并用 RRF 融合"""

    # 1. 加载 chunks（命中缓存就秒加载）
    chunks = load_chunks(docs_dir)

    # 2. BM25
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    # 3. 向量检索器
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        cache_folder="./models",           # 模型也缓存到本地
    )
    vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # 4. RRF 融合
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]
    )
    return hybrid_retriever