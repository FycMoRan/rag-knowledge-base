import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_classic.chains import RetrievalQA

# 读取 .env 里的 DEEPSEEK_API_KEY
load_dotenv()

# ========== 1. 加载已有的向量数据库 ==========
print("正在加载向量库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ========== 2. 创建检索器 ==========
# k=3 表示每次检索最相关的 3 个片段
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== 3. 初始化 DeepSeek 大模型 ==========
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.2
)

# ========== 4. 构建问答链 ==========
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ========== 5. 进入问答循环 ==========
print("=" * 50)
print("进入问答模式，输入 q 退出")
print("=" * 50)

while True:
    question = input("\n请输入你的问题（输入 q 退出）: ")
    if question.strip().lower() in ("q", "quit", "exit"):
        print("再见！")
        break

    result = qa_chain.invoke({"query": question})
    print("\n回答:", result["result"])
    print("\n来源文档:")
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        print(f"- [来源: {source} 第{page}页] {doc.page_content[:100]}...")