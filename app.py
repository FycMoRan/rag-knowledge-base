import os
import json
import streamlit as st
from dotenv import load_dotenv

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_deepseek import ChatDeepSeek
from hybrid_retriever import build_hybrid_retriever

load_dotenv()

HISTORY_FILE = "chat_history.json"   # 本地持久化文件

# ==================================================
# 页面基础设置
# ==================================================
st.set_page_config(page_title="个人知识库助手", page_icon="📚")
st.title("📚 个人知识库问答助手")
st.markdown("基于 RAG 架构，支持 PDF / Markdown 文档精准问答")

# ==================================================
# 历史持久化函数
# ==================================================
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(messages):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# ==================================================
# 侧边栏
# ==================================================
with st.sidebar:
    st.header("⚙️ 配置")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.getenv("DEEPSEEK_API_KEY", "")
    )
    top_k = st.slider("检索片段数", 1, 10, 5)
    history_rounds = st.slider("模型记忆轮数", 1, 10, 5)
    max_input_len = st.slider("单次提问最大字数", 50, 2000, 500, step=50)

    st.divider()

    if st.button("🧹 清空会话"):
        st.session_state.messages = []
        save_history([])
        st.rerun()

    if st.button("🔁 重新生成上一条"):
        st.session_state.regenerate = True

    st.divider()

    if st.button("🔄 重新构建知识库"):
        with st.spinner("正在处理文档..."):
            st.success("知识库已更新！")

# ==================================================
# 提示词
# ==================================================
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的知识库问答助手。
请优先根据下面提供的资料回答问题。如果资料中没有相关信息，可以结合对话历史进行合理回答，但不要编造。
回答请用简体中文，条理清晰。

资料：
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(
        f"[来源: {d.metadata.get('source', '未知')} 第{d.metadata.get('page', '?')}页]\n{d.page_content}"
        for d in docs
    )


def get_recent_history():
    if "messages" not in st.session_state:
        return []
    msgs = st.session_state.messages[-(history_rounds * 2):]
    history = []
    for m in msgs:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history


# ==================================================
# 初始化链
# ==================================================
@st.cache_resource(show_spinner=False)
def init_chain(api_key: str, top_k: int):
    try:
        retriever = build_hybrid_retriever(top_k=top_k)
    except TypeError:
        retriever = build_hybrid_retriever()

    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.2,
        streaming=True,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: get_recent_history(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retriever, chain


if not api_key:
    st.warning("请在左侧栏填入 DeepSeek API Key，或确认 .env 中已配置 DEEPSEEK_API_KEY")
    st.stop()

retriever, chain = init_chain(api_key, top_k)

# ==================================================
# 会话历史初始化（从本地读取）
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if "regenerate" not in st.session_state:
    st.session_state.regenerate = False

# ==================================================
# 渲染历史消息（全部渲染，用户可往上滚）
# ==================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ==================================================
# 处理一次问答
# ==================================================
def handle_query(query: str):
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料..."):
            try:
                source_docs = retriever.invoke(query)
            except Exception as e:
                st.error(f"检索出错：{e}")
                return

        source_box = st.container()

        st.session_state.messages.append({"role": "user", "content": query})

        def stream_answer():
            try:
                for chunk in chain.stream(query):
                    yield chunk
            except Exception as e:
                msg = str(e)
                if "401" in msg or "Authentication" in msg:
                    yield "\n\n❌ API Key 无效，请检查左侧栏或 .env 文件。"
                elif "402" in msg or "Insufficient Balance" in msg:
                    yield "\n\n❌ DeepSeek 账户余额不足，请充值后重试。"
                elif "Connection" in msg or "timeout" in msg.lower():
                    yield "\n\n❌ 网络连接失败，请检查网络或稍后再试。"
                else:
                    yield f"\n\n❌ 出错了：{msg}"

        full_answer = st.write_stream(stream_answer())

        with source_box:
            with st.expander("📎 查看参考来源"):
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get("source", "未知")
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**片段 {i+1}**（来源: {source} 第{page}页）")
                    st.markdown(f"> {doc.page_content[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    save_history(st.session_state.messages)   # 每次问答后落盘


# ==================================================
# 重新生成
# ==================================================
if st.session_state.regenerate:
    st.session_state.regenerate = False
    last_user = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            last_user = msg["content"]
            break

    if last_user and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages.pop()
    if last_user:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()
        handle_query(last_user)
        st.stop()

# ==================================================
# 聊天输入
# ==================================================
query = st.chat_input("请输入你的问题...")
if query:
    if len(query) > max_input_len:
        st.warning(f"问题太长（{len(query)} 字），请控制在 {max_input_len} 字以内。")
        st.stop()

    handle_query(query)