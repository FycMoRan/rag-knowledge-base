import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("开始验证各包...")

import langchain
print("langchain OK")

import langchain_community
print("langchain-community OK")

import langchain_text_splitters
print("langchain-text-splitters OK")

import chromadb
print("chromadb OK")

from sentence_transformers import SentenceTransformer
print("sentence-transformers OK")

import pypdf
print("pypdf OK")

import docx
print("python-docx OK")

from rank_bm25 import BM25Okapi
print("rank-bm25 OK")

import streamlit
print("streamlit OK")

import dotenv
print("python-dotenv OK")

print("\n所有包导入成功 ✅")
