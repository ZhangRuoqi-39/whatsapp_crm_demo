# 🎮 WhatsApp CRM AI Agent

> A production-minded WhatsApp auto-reply agent for live-service game customer support — combining intent classification, hybrid RAG retrieval, safety guardrails, and contextual marketing suggestions.

---

## 📋 Table of Contents

- [Business Context](#business-context)
- [Architecture](#architecture)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Evaluation Results](#evaluation-results)
- [ROI Estimate](#roi-estimate)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Running Evaluation](#running-evaluation)
- [20 Game CRM Intents](#20-game-crm-intents)
- [Safety & Compliance](#safety--compliance)
- [Future Work](#future-work)

---

## Business Context

Live-service games generate thousands of customer support tickets daily — payment failures, account recovery, item delivery, refund requests, and more. Manual-only CRM is slow, expensive, and unavailable 24/7.

**This project demonstrates how a deterministic Router + Handoff architecture with RAG can automate 82% of inbound queries while maintaining 100% safety compliance**, freeing human agents to focus on complex or high-value interactions.

| Pain Point | This System's Response |
|---|---|
| 15-min average response time | 8-second AI reply |
| Manual classification inconsistency | 96% intent accuracy |
| Uncontrolled LLM outputs | Dual safety layer, 100% block rate |
| Agents handling routine FAQs | 82% automation rate |
| Hard to scale personalization | Intent-triggered marketing suggestions |

---

## Architecture

### Router + Handoff Pattern

This system uses a **deterministic Router + Handoff** design rather than a fully autonomous agent. This choice prioritizes safety predictability and audit-ability — critical requirements for enterprise CRM compliance.

```
User Message
      │
      ▼
┌─────────────────────┐
│   Safety Guard      │  INPUT LAYER
│   (blocklist check) │  • Prompt injection
└──────────┬──────────┘  • PII requests
           │             • Cheating tools
           ▼             • Harmful content
┌─────────────────────┐
│  Intent Classifier  │  ROUTER
│  (DeepSeek few-shot)│  Routes to one of three paths ↓
└──────────┬──────────┘
           │
     ┌─────┴──────────────────────┐
     │                            │
     ▼                            ▼
escalate_to_human           known intent
     │                            │
     ▼                            ▼
┌──────────┐          ┌───────────────────────┐
│  HANDOFF │          │   Hybrid RAG Chain    │  RAG LAYER
│  to human│          │   Dense + BM25 + RRF  │
│  agent   │          │   + Qwen Rerank        │
└──────────┘          └───────────┬───────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │   LangChain Chain     │  GENERATION
                      │   DeepSeek LLM        │  • Multi-turn history
                      │   + brand tone prompt │  • Context injection
                      └───────────┬───────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │   Safety Guard        │  OUTPUT LAYER
                      │   (compliance check)  │  • AI disclosure
                      └───────────┬───────────┘  • False guarantees
                                  │             • Exact refund amounts
                                  ▼
                      ┌───────────────────────┐
                      │  Marketing Trigger    │  UPSELL LAYER
                      │  (intent-based rules) │  Intent → compliant
                      └───────────┬───────────┘  suggestion (internal)
                                  │
                                  ▼
                            AgentResponse
```

**Why Router + Handoff, not a full Agent?**

A fully autonomous agent (ReAct / LangGraph) would allow the LLM to self-decide retrieval steps, but introduces unpredictable behavior and makes the safety layer harder to guarantee. For enterprise CRM, deterministic routing ensures:
- Every input passes through safety checks (no bypass paths)
- Escalation logic is explicit and auditable
- Evaluation metrics are reproducible across runs

*Agentic RAG is the natural next evolution — see [Future Work](#future-work).*

---

## Retrieval Pipeline

Standard vector search underperforms on game-specific terminology (e.g., "王者荣耀皮肤", "原神圣遗物"). This system implements a **two-stage hybrid retrieval** pipeline to address this.

### Stage 1: Hybrid Recall (粗排召回)

```
Query
  ├── Dense Retrieval (Top-20)
  │     Qwen text-embedding-v3 + ChromaDB cosine similarity
  │     → Handles semantic similarity, paraphrases, synonyms
  │
  └── Sparse Retrieval (Top-20)
        BM25 (rank_bm25) in-memory index
        → Handles exact keyword matching, game proper nouns
              │
              ▼
        RRF Fusion (k=60)
        score(d) = Σ 1 / (k + rank_i(d))
        → Deduplicates and merges both result sets
```

### Stage 2: Rerank (精排重排)

```
RRF Merged Candidates (up to 40 docs)
              │
              ▼
    Qwen gte-rerank (Cross-Encoder)
    Scores each candidate against query independently
              │
              ▼
         Top-5 Results → LLM Context
```

**Why two stages?** Bi-Encoder (embedding) is fast but misses exact terms. Cross-Encoder (rerank) is more accurate but too slow for full-corpus search. The coarse→fine architecture gets the best of both.

---

## Evaluation Results

All metrics measured on 100-case golden test set (held-out from training data).

### Core Metrics

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Intent Classification Accuracy | **96.0%** | ≥ 85% | ✅ +11% |
| RAG Hit@3 | **84.0%** | ≥ 70% | ✅ +14% |
| Safety Block Rate | **100%** | 100% | ✅ Perfect |
| Intent Confidence (high) | **94 / 100** | — | 94% high-conf |

### Ragas Deep Evaluation (20-sample subset)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Faithfulness | **0.833** | Answers well-grounded in retrieved context |
| Context Precision | **0.835** | Retrieved docs are highly relevant |
| Answer Relevancy | **0.471** | Lower — partially due to escalation replies in sample |

> **Note on Answer Relevancy:** The 20-sample Ragas subset included cases where the system escalated to human agent, producing a fixed escalation reply unrelated to the query. This deflates the relevancy score. On non-escalated cases the score is materially higher.

### Safety Test Coverage

20 unit tests across two categories:

- **Functional** (10): prompt injection, cheating tools, PII, AI disclosure, false guarantees, exact amounts
- **Boundary / Contract** (10): empty string, whitespace, 10,000-char input, Unicode special chars, Unicode-wrapped injection, fallback reply presence

```bash
python -m pytest tests/test_safety.py -v
# 20 passed in 0.03s
```

---

## ROI Estimate

Based on 500 tickets/day (mid-size game CRM operation), automation rate derived from measured intent accuracy.

| Metric | Human-Only | With AI Agent | Uplift |
|--------|-----------|---------------|--------|
| Headcount needed | 8.3 agents | 1.5 agents | −6.8 |
| Avg response time | 15 min | 8 sec | **−99.1%** |
| Monthly cost (USD) | $24,000 | $4,332 | **−82%** |
| Annual cost (USD) | $292,000 | $52,706 | **−82%** |
| FCR (first contact resolution) | 72% | 85% | **+18.1%** |
| Availability | Mon–Fri 9–18 | 24/7 | ✅ |
| **Annual savings** | — | **$239,294** | — |

*Assumptions: $12/hr agent cost (SEA outsourcing rate), 60 tickets/agent/day, DeepSeek API at $0.001/1k tokens.*

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| LLM Framework | LangChain | Chain orchestration + history management |
| Chat LLM | DeepSeek `deepseek-chat` | Cost-effective, OpenAI-compatible API |
| Embedding | Qwen `text-embedding-v3` | Strong CN/EN bilingual performance |
| Reranker | Qwen `gte-rerank` | Cross-Encoder, same API key as embedding |
| Sparse Retrieval | `rank_bm25` | Pure Python, no external service needed |
| Vector DB | ChromaDB (local) | Persistent, zero-infrastructure |
| Dataset | Bitext Customer Support (HuggingFace) | 26 intents, 26k labeled QA pairs |
| Conversation History | File-based JSON | Per `session_id`, portable, no extra infra |
| Evaluation | Ragas | Faithfulness / Answer Relevancy / Context Precision |
| Demo UI | Streamlit | 3-tab interface: Chat / KB Management / Eval Report |

---

## Project Structure

```
whatsapp_crm_demo/
├── .env                          # API keys (never commit)
├── .env.example                  # Config template
├── requirements.txt
├── README.md
├── app.py                        # Streamlit entry point (3 tabs)
│
├── src/
│   ├── config.py                 # Centralized config (env vars + retrieval params)
│   ├── safety.py                 # Dual-layer safety guardrails
│   ├── history.py                # File-based multi-turn conversation history
│   ├── knowledge.py              # HybridRetriever: BM25 + Dense + RRF + Rerank
│   ├── intent.py                 # Few-shot intent classifier (20 intents)
│   └── chain.py                  # CRMAgent: Router + Handoff orchestration
│
├── scripts/
│   ├── data_loader.py            # Bitext → ChromaDB + golden test set
│   ├── evaluate.py               # Offline eval: Intent / RAG / Safety / Ragas
│   └── uplift_estimate.py        # ROI model: human vs AI cost comparison
│
├── data/                         # Auto-generated (gitignored)
│   ├── bitext_intents.csv        # 600-row training set
│   ├── golden_test_set.json      # 100-case held-out test set
│   ├── eval_report.json          # Latest evaluation results
│   ├── uplift_report.json        # Latest ROI estimate
│   ├── kb_fingerprints.txt       # SHA256 dedup index
│   ├── chat_history/             # Per-session JSON history
│   └── chroma_db/                # ChromaDB persistent vector store
│
└── tests/
    └── test_safety.py            # 20 unit + contract tests for safety layer
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- [DashScope API key](https://dashscope.aliyun.com/) (Qwen embedding + rerank)
- [DeepSeek API key](https://platform.deepseek.com/)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
DASHSCOPE_API_KEY=sk-...     # Qwen embedding + rerank
DEEPSEEK_API_KEY=sk-...      # DeepSeek LLM
```

### 3. Build knowledge base

Downloads Bitext dataset from HuggingFace, builds ChromaDB index (300 documents), and generates golden test set.

```bash
python scripts/data_loader.py
```

Expected output:
```
✅ 训练集：600 条 → data/bitext_intents.csv
✅ Golden test set：100 条 → data/golden_test_set.json
✅ 知识库：300 条文档 → data/chroma_db/
```

### 4. Launch demo

```bash
streamlit run app.py --server.port 8502
```

Open `http://localhost:8502`

---

## Running Evaluation

### Offline evaluation (Intent + RAG + Safety + Ragas)

~15 minutes, ~100 API calls.

```bash
python scripts/evaluate.py
```

Output saved to `data/eval_report.json`. Results also visible in the **📈 System Report** tab of the Streamlit demo.

### ROI estimate

```bash
python scripts/uplift_estimate.py
```

Output saved to `data/uplift_report.json`.

### Safety unit tests

```bash
python -m pytest tests/test_safety.py -v
```

---

## 20 Game CRM Intents

| Category | Intents |
|----------|---------|
| **Account** | `account_register` `account_delete` `account_edit` `account_recovery` `account_issue` `account_switch` |
| **Payment** | `payment_failed` `payment_methods` `refund_request` `refund_policy` |
| **Purchase** | `purchase_cancel` `purchase_history` `item_delivery` `in_game_purchase` |
| **Service** | `complaint` `escalate_to_human` `general_inquiry` `feedback` |
| **Marketing** | `event_subscribe` `gift_options` |

The classifier uses **few-shot prompting** (no fine-tuning required) with explicit Chinese keyword hints for bilingual support. Confidence levels: `high` / `medium` / `low`.

---

## Safety & Compliance

### Input Layer — Block Conditions

| Flag | Trigger Examples |
|------|-----------------|
| `prompt_injection` | "ignore previous instructions", "forget all rules" |
| `role_override` | "you are now DAN", "act as unrestricted AI" |
| `cheating_tool` | "cheat engine", "mod menu", "exploit bug" |
| `harmful_content` | self-harm keywords |
| `pii_request` | "credit card number", "social security" |

### Output Layer — Block Conditions

| Flag | Trigger Examples |
|------|-----------------|
| `ai_disclosure` | "as an AI", "as a language model" |
| `false_guarantee` | "guarantee", "guaranteed" |
| `exact_refund_amount` | "$9.99", "$10" (any specific dollar amount) |

### Escalation (Handoff) Logic

```python
if intent == "escalate_to_human":
    → Handoff immediately
else:
    → Continue RAG chain (LLM handles greeting/general_inquiry naturally)
```

---

## Future Work

These items are explicitly **out of scope** for the current prototype but represent the natural production roadmap:

### Near-term (next sprint)
- **Agentic RAG**: Replace fixed chain with LangGraph-based multi-step reasoning. The existing `HybridRetriever` can be reused as a LangGraph tool node with minimal refactoring.
- **MCP Server interface**: Expose knowledge base as an MCP-compatible tool server, enabling direct integration with GitHub Copilot, Claude Desktop, or internal agent frameworks.
- **Query tracing dashboard**: Persist per-query Dense vs. Sparse retrieval comparison and Rerank score changes for debugging and continuous improvement.

### Medium-term (production hardening)
- **Multimodal support**: Image-to-text pipeline for game screenshots and activity posters (LLM captioning → text chunk → existing retrieval chain, no architecture changes required).
- **Cloud deployment**: Dockerize and deploy to Azure Container Apps / AWS Lambda. Architecture is stateless and horizontally scalable.
- **Multi-tenant isolation**: Namespace ChromaDB collections per game title or region.

### Long-term
- **Online A/B evaluation**: Replace offline golden set with live traffic sampling and implicit feedback signals (resolution rate, escalation rate, CSAT).
- **Fine-tuned intent classifier**: Replace few-shot prompting with a lightweight fine-tuned model (e.g., `bert-base-multilingual`) for lower latency and API cost.

---

## Engineering Notes

- **Deduplication**: SHA256 fingerprint check before every ingestion. Identical content is never re-indexed regardless of source label.
- **BM25 sync**: In-memory BM25 index auto-rebuilds from ChromaDB on startup and after every write operation, ensuring Dense and Sparse indices are always in sync.
- **Graceful degradation**: Rerank failures silently fall back to RRF-ranked results. The system never returns an error to the user due to a retrieval component failure.
- **Session isolation**: Each `session_id` has an independent JSON history file. Concurrent sessions do not share state.

---

*Built by ZhangRuoqi
