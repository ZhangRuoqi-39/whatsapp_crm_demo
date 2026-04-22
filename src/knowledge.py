"""
knowledge.py
------------
知识库管理：文档上传、分割、向量化、Hybrid Search检索、Rerank重排。

检索架构（两阶段）：
  ┌─────────────────────────────────────────────┐
  │  粗排召回（Hybrid Search）                    │
  │  Dense: 通义千问Embedding + ChromaDB          │
  │    +                                         │
  │  Sparse: BM25 关键词检索 (rank_bm25)          │
  │    ↓                                         │
  │  RRF融合 (Reciprocal Rank Fusion)             │
  └─────────────────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────────┐
  │  精排重排（Rerank）                           │
  │  通义千问 gte-rerank Cross-Encoder            │
  └─────────────────────────────────────────────┘
                    ↓
             Top-K 最终结果

对外接口：
    kb = KnowledgeBase()
    kb.add_texts(text, source)    # 添加文本
    kb.add_file(file_path)        # 添加文件 (.txt/.pdf)
    kb.add_documents(docs)        # 添加LangChain Document列表
    kb.search(query, top_k)       # Hybrid Search + Rerank
    kb.get_retriever()            # 返回LangChain Retriever（供chain使用）
    kb.count()                    # 知识库文档数
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

import dashscope
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DASHSCOPE_API_KEY, EMBEDDING_MODEL, RERANK_MODEL,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS,
    DENSE_TOP_K, SPARSE_TOP_K, RRF_K, RERANK_TOP_K,
    FINGERPRINT_FILE,
)


# ── SHA256去重工具（比MD5更严谨）────────────────────────────
def _get_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _fingerprint_exists(fp: str) -> bool:
    if not os.path.exists(FINGERPRINT_FILE):
        return False
    return any(line.strip() == fp for line in open(FINGERPRINT_FILE, encoding="utf-8"))

def _save_fingerprint(fp: str) -> None:
    Path(FINGERPRINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(FINGERPRINT_FILE, "a", encoding="utf-8") as f:
        f.write(fp + "\n")


# ── RRF融合算法 ──────────────────────────────────────────────
def _rrf_fusion(
    dense_docs:  list[Document],
    sparse_docs: list[Document],
    k: int = RRF_K,
) -> list[Document]:
    """
    Reciprocal Rank Fusion：合并Dense和Sparse两路检索结果。
    score(d) = Σ 1 / (k + rank_i(d))
    返回按RRF分数降序排列的去重文档列表。
    """
    scores: dict[str, float]   = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        uid = doc.page_content[:100]          # 用前100字符作为唯一标识
        scores[uid]  = scores.get(uid, 0) + 1 / (k + rank)
        doc_map[uid] = doc

    for rank, doc in enumerate(sparse_docs, start=1):
        uid = doc.page_content[:100]
        scores[uid]  = scores.get(uid, 0) + 1 / (k + rank)
        doc_map[uid] = doc

    sorted_uids = sorted(scores, key=lambda u: scores[u], reverse=True)
    return [doc_map[uid] for uid in sorted_uids]


# ── 通义千问Rerank ───────────────────────────────────────────
def _rerank(query: str, docs: list[Document], top_k: int = RERANK_TOP_K) -> list[Document]:
    """
    调用通义千问 gte-rerank Cross-Encoder 对候选文档重排。
    返回rerank后top_k个文档（按相关性降序）。
    """
    if not docs:
        return []

    dashscope.api_key = DASHSCOPE_API_KEY
    passages = [doc.page_content for doc in docs]

    try:
        resp = dashscope.TextReRank.call(
            model=RERANK_MODEL,
            query=query,
            documents=passages,
            top_n=min(top_k, len(docs)),
            return_documents=False,
        )
        if resp.status_code != 200:
            # Rerank失败时降级：直接返回RRF结果的前top_k
            print(f"⚠️  Rerank失败（{resp.status_code}），降级为RRF结果")
            return docs[:top_k]

        # 按rerank分数重排，映射回原Document对象
        ranked = sorted(resp.output.results, key=lambda x: x.relevance_score, reverse=True)
        return [docs[item.index] for item in ranked]

    except Exception as e:
        print(f"⚠️  Rerank异常（{e}），降级为RRF结果")
        return docs[:top_k]


# ── HybridRetriever：LangChain Retriever接口 ─────────────────
class HybridRetriever(BaseRetriever):
    """
    实现LangChain BaseRetriever接口的混合检索器。
    内部执行：BM25召回 + Dense召回 → RRF融合 → Rerank精排
    可直接插入LangChain chain（与普通Retriever接口完全兼容）。
    """
    # LangChain要求用类型注解声明字段（Pydantic模型）
    vectorstore:   object
    bm25_index:    Optional[object] = None
    bm25_docs:     list = []
    dense_top_k:   int = DENSE_TOP_K
    sparse_top_k:  int = SPARSE_TOP_K
    rerank_top_k:  int = RERANK_TOP_K
    enable_rerank: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:

        # ── Step 1: Dense检索 ────────────────────────────────
        dense_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.dense_top_k},
        )
        dense_docs = dense_retriever.invoke(query)

        # ── Step 2: BM25稀疏检索 ─────────────────────────────
        sparse_docs = []
        if self.bm25_index and self.bm25_docs:
            tokens     = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokens)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[: self.sparse_top_k]
            sparse_docs = [self.bm25_docs[i] for i in top_indices if bm25_scores[i] > 0]

        # ── Step 3: RRF融合 ──────────────────────────────────
        fused_docs = _rrf_fusion(dense_docs, sparse_docs)

        # ── Step 4: Rerank精排 ───────────────────────────────
        if self.enable_rerank and fused_docs:
            return _rerank(query, fused_docs, top_k=self.rerank_top_k)
        else:
            return fused_docs[: self.rerank_top_k]


# ── KnowledgeBase 主类 ───────────────────────────────────────
class KnowledgeBase:
    """
    知识库管理类，对外暴露文档管理与检索接口。
    内部维护：ChromaDB（Dense） + BM25索引（Sparse） + Reranker
    """

    def __init__(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        # Dense向量存储
        try:
            import streamlit as st
            _key = st.secrets.get("DASHSCOPE_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
        except Exception:
            _key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._embedding = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=_key,
        )
        self._vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=self._embedding,
            persist_directory=CHROMA_PERSIST_DIR,
        )

        # 文本分割器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
        )

        # BM25索引（从ChromaDB现有数据初始化）
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs:  list[Document]      = []
        self._rebuild_bm25()

        # 若ChromaDB为空，自动从knowledge_base/文件夹重建
        if self.count() == 0:
            self._auto_load_knowledge_base()

    # ── 私有方法 ─────────────────────────────────────────────
    def _auto_load_knowledge_base(self) -> None:
        """ChromaDB为空时，自动读取knowledge_base/下所有txt文件重建知识库。"""
        kb_dir = Path(__file__).parent.parent / "knowledge_base"
        if not kb_dir.exists():
            return
        for txt_file in sorted(kb_dir.glob("*.txt")):
            print(f"[KnowledgeBase] 自动加载：{txt_file.name}")
            result = self.add_file(str(txt_file))
            print(result)

    def _rebuild_bm25(self) -> None:
        """从ChromaDB重新构建BM25内存索引（每次启动或写入后刷新）。"""
        try:
            result = self._vectorstore.get()
            if not result or not result.get("documents"):
                self._bm25_index = None
                self._bm25_docs  = []
                return

            texts = result["documents"]
            metas = result.get("metadatas", [{}] * len(texts))
            self._bm25_docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metas)
            ]
            tokenized = [t.lower().split() for t in texts]
            self._bm25_index = BM25Okapi(tokenized)
        except Exception as e:
            print(f"⚠️  BM25索引构建失败：{e}")
            self._bm25_index = None
            self._bm25_docs  = []

    def _add_to_vectorstore(self, docs: list[Document]) -> None:
        """写入ChromaDB并刷新BM25索引。"""
        self._vectorstore.add_documents(docs)
        self._rebuild_bm25()    # 保持BM25与ChromaDB同步

    # ── 公开接口：文档管理 ───────────────────────────────────
    def add_texts(self, text: str, source: str = "manual") -> str:
        """将纯文本字符串分割后添加到知识库（SHA256去重）。"""
        fp = _get_sha256(text)
        if _fingerprint_exists(fp):
            return f"⚠️ [跳过] 内容已存在知识库中（{source}）"

        import hashlib as _hl
        doc_id = _hl.md5(text[:64].encode()).hexdigest()[:8]
        docs = self._splitter.create_documents(
            texts=[text],
            metadatas=[{"source": source, "type": "text", "doc_id": doc_id}],
        )
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["total_chunks"] = len(docs)
        self._add_to_vectorstore(docs)
        _save_fingerprint(fp)
        return f"✅ [成功] 已添加 {len(docs)} 个文本块到知识库（{source}）"

    def add_file(self, file_path: str) -> str:
        """从本地文件路径加载并添加到知识库，支持 .txt 和 .pdf。"""
        path   = Path(file_path)
        if not path.exists():
            return f"❌ 文件不存在：{file_path}"

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            return f"❌ 不支持的格式：{suffix}（仅支持 .txt / .pdf）"

        raw_docs  = loader.load()
        full_text = " ".join(d.page_content for d in raw_docs)

        fp = _get_sha256(full_text)
        if _fingerprint_exists(fp):
            return f"⚠️ [跳过] 文件已存在知识库中（{path.name}）"

        docs = self._splitter.split_documents(raw_docs)
        import hashlib as _hl
        doc_id = _hl.md5(path.name.encode()).hexdigest()[:8]
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source":       path.name,
                "type":         suffix.lstrip("."),
                "doc_id":       doc_id,
                "chunk_index":  i,
                "total_chunks": len(docs),
            })

        self._add_to_vectorstore(docs)
        _save_fingerprint(fp)
        return f"✅ [成功] 已添加 {len(docs)} 个文本块到知识库（{path.name}）"

    def add_documents(self, docs: list[Document]) -> str:
        """直接添加LangChain Document列表（供data_loader批量调用）。"""
        if not docs:
            return "⚠️ 无文档可添加"
        self._add_to_vectorstore(docs)
        return f"✅ 已添加 {len(docs)} 个文档块"

    # ── 公开接口：检索 ───────────────────────────────────────
    def search(self, query: str, top_k: int = RERANK_TOP_K) -> list[Document]:
        """
        完整两阶段检索：Hybrid Search → Rerank。
        直接调用时使用此方法（如evaluate.py）。
        """
        retriever = self.get_retriever(top_k=top_k)
        return retriever.invoke(query)

    def get_retriever(self, top_k: int = RERANK_TOP_K) -> HybridRetriever:
        """
        返回HybridRetriever实例，兼容LangChain chain接口。
        内部执行：BM25 + Dense → RRF → Rerank
        """
        return HybridRetriever(
            vectorstore=self._vectorstore,
            bm25_index=self._bm25_index,
            bm25_docs=self._bm25_docs,
            dense_top_k=DENSE_TOP_K,
            sparse_top_k=SPARSE_TOP_K,
            rerank_top_k=top_k,
            enable_rerank=True,
        )

    def count(self) -> int:
        """返回当前知识库文档数。"""
        return self._vectorstore._collection.count()
