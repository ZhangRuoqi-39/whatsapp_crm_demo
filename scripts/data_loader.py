"""
data_loader.py
--------------
从HuggingFace加载Bitext数据集，构建：
  - data/bitext_intents.csv     : Intent分类训练数据
  - data/golden_test_set.json   : 离线评估golden set
  - 知识库（通过KnowledgeBase写入ChromaDB）
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from langchain_core.documents import Document
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import INTENT_TAXONOMY
from knowledge import KnowledgeBase

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

SELECTED_INTENTS   = list(INTENT_TAXONOMY.keys())
N_TRAIN_PER_INTENT = 30
N_TEST_PER_INTENT  = 5
N_KB_PER_INTENT    = 15   # 每个intent存入知识库的QA对数量


def main():
    print("=" * 55)
    print("  WhatsApp CRM — 数据加载 & 知识库构建")
    print("=" * 55)

    # 1. 加载Bitext
    print("📥 加载 Bitext 数据集...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        trust_remote_code=True,
    )
    df = ds["train"].to_pandas()
    df = df[df["intent"].isin(SELECTED_INTENTS)].copy()
    df = df.rename(columns={"instruction": "user_message", "response": "expected_response"})
    df["game_label"] = df["intent"].map(lambda x: INTENT_TAXONOMY[x]["game_label"])
    df["zh_label"]   = df["intent"].map(lambda x: INTENT_TAXONOMY[x]["zh"])

    # 2. 训练集
    train_rows = []
    for intent in SELECTED_INTENTS:
        sub = df[df["intent"] == intent]
        train_rows.append(sub.sample(n=min(N_TRAIN_PER_INTENT, len(sub)), random_state=42))
    train_df = pd.concat(train_rows).reset_index(drop=True)
    train_df[["user_message", "intent", "game_label", "zh_label", "expected_response"]].to_csv(
        DATA_DIR / "bitext_intents.csv", index=False, encoding="utf-8"
    )
    print(f"  ✅ 训练集：{len(train_df)} 条 → data/bitext_intents.csv")

    # 3. Golden test set（不与训练集重叠）
    golden = []
    train_ids = set(train_df.index)
    for intent in SELECTED_INTENTS:
        sub = df[df["intent"] == intent]
        remaining = sub[~sub.index.isin(train_ids)]
        sample = remaining.sample(n=min(N_TEST_PER_INTENT, len(remaining)), random_state=99)
        for _, row in sample.iterrows():
            golden.append({
                "query":            row["user_message"],
                "expected_intent":  row["game_label"],
                "expected_snippet": row["expected_response"][:200],
            })
    with open(DATA_DIR / "golden_test_set.json", "w", encoding="utf-8") as f:
        json.dump({"test_cases": golden, "total": len(golden)}, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Golden test set：{len(golden)} 条 → data/golden_test_set.json")

    # 4. 构建知识库（LangChain Documents → ChromaDB）
    print("🔨 构建知识库...")
    kb   = KnowledgeBase()
    docs = []
    for intent in SELECTED_INTENTS:
        sub = train_df[train_df["intent"] == intent].head(N_KB_PER_INTENT)
        for _, row in sub.iterrows():
            docs.append(Document(
                page_content=f"Q: {row['user_message']}\nA: {row['expected_response']}",
                metadata={
                    "intent":   row["game_label"],
                    "zh_label": row["zh_label"],
                    "source":   "bitext_customer_support",
                },
            ))

    # 批量写入（避免一次性太多）
    batch_size = 50
    for i in tqdm(range(0, len(docs), batch_size), desc="写入ChromaDB"):
        kb.add_documents(docs[i: i + batch_size])

    print(f"  ✅ 知识库共 {kb.count()} 条文档 → ChromaDB")
    print("\n✅ 数据准备完成！下一步：streamlit run app.py")


if __name__ == "__main__":
    main()
