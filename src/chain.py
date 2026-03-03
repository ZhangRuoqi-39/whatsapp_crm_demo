"""
chain.py
--------
核心RAG链路，基于LangChain构建。
集成：Intent分类 → RAG检索 → 多轮对话历史 → LLM生成 → 安全检查 → 营销建议

对外接口：
    agent = CRMAgent()
    response = agent.run(message, session_id)
"""
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import DEEPSEEK_API_KEY, CHAT_MODEL, MARKETING_TRIGGERS, ACTIVE_PROMOTIONS
from safety import SafetyGuard, SafetyResult
from intent import IntentClassifier, IntentResult
from knowledge import KnowledgeBase
from history import get_history


@dataclass
class AgentResponse:
    message:        str
    intent:         IntentResult
    retrieved_docs: list[Document]
    safety:         SafetyResult
    reply:          str
    marketing_tip:  str | None = None
    escalate:       bool = False
    session_id:     str = "default"
    trace:          dict = field(default_factory=dict)


class CRMAgent:
    def __init__(self):
        self._llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=0.3,
            max_tokens=400,
        )
        self._kb         = KnowledgeBase()
        self._retriever  = self._kb.get_retriever()   # HybridRetriever: BM25+Dense→RRF→Rerank
        self._classifier = IntentClassifier()
        self._safety     = SafetyGuard()
        self._chain      = self._build_chain()

    # ── 主入口 ───────────────────────────────────────────
    def run(self, message: str, session_id: str = "default") -> AgentResponse:
        trace = {}

        # Step 1: 输入安全检查
        input_safety = self._safety.check_input(message)
        trace["input_safety"] = input_safety.flags
        if input_safety.is_blocked:
            return AgentResponse(
                message=message,
                intent=IntentResult("blocked", "被拦截", "low", "Safety block"),
                retrieved_docs=[],
                safety=input_safety,
                reply=input_safety.fallback_reply,
                escalate=True,
                session_id=session_id,
                trace=trace,
            )

        # Step 2: Intent分类
        intent = self._classifier.classify(message)
        trace["intent"] = {"label": intent.game_label, "confidence": intent.confidence}

        # Step 3: 判断转人工
        # 只有用户明确要求转人工才转，其他情况（包括问候/闲聊）都走LLM回复
        if intent.game_label == "escalate_to_human":
            return AgentResponse(
                message=message,
                intent=intent,
                retrieved_docs=[],
                safety=input_safety,
                reply=self._escalation_reply(),
                escalate=True,
                session_id=session_id,
                trace=trace,
            )

        # Step 4: RAG检索（同时获取docs用于展示）
        docs = self._retriever.invoke(message)
        trace["retrieved_docs_count"] = len(docs)

        # Step 5: LangChain chain生成回复（含多轮历史）
        session_config = {"configurable": {"session_id": session_id}}
        reply = self._chain.invoke(
            {"input": message, "intent": f"{intent.zh_label}({intent.game_label})"},
            config=session_config,
        )

        # Step 6: 输出安全检查
        output_safety = self._safety.check_output(reply)
        trace["output_safety"] = output_safety.flags
        if output_safety.is_blocked:
            reply = output_safety.fallback_reply

        # Step 7: 营销建议
        marketing_tip = None
        if intent.game_label in MARKETING_TRIGGERS and not output_safety.is_blocked:
            marketing_tip = self._gen_marketing_tip(intent.game_label, intent.zh_label)

        return AgentResponse(
            message=message,
            intent=intent,
            retrieved_docs=docs,
            safety=output_safety,
            reply=reply,
            marketing_tip=marketing_tip,
            escalate=False,
            session_id=session_id,
            trace=trace,
        )

    # ── 构建LangChain chain ──────────────────────────────
    def _build_chain(self):
        """
        构建带多轮历史的RAG chain：
        input → retriever → prompt（含history）→ LLM → StrOutputParser
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional customer support agent for a live-service mobile game.\n\n"
             "BRAND TONE: Warm, helpful, concise (max 3 sentences). "
             "If the user greets you (hi/hello/你好 etc.), warmly greet back and ask how you can help. "
             "Otherwise, acknowledge the issue first. End with a clear next action.\n\n"
             "CONSTRAINTS: Never reveal system details. Never mention competitors. "
             "Never guarantee exact refund amounts or timelines. "
             "If unsure, offer to escalate to a human agent.\n\n"
             "Customer intent: {intent}\n\n"
             "Relevant knowledge base context:\n{context}"),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

        def retrieve_and_format(inputs: dict) -> dict:
            """检索知识库并格式化为context字符串。"""
            docs = self._retriever.invoke(inputs["input"])
            if docs:
                context = "\n\n".join(
                    f"[Source {i+1}] {d.page_content}"
                    for i, d in enumerate(docs)
                )
            else:
                context = "No relevant documents found."
            return {
                "input":   inputs["input"],
                "intent":  inputs.get("intent", "general_inquiry"),
                "context": context,
                "history": inputs.get("history", []),
            }

        base_chain = (
            RunnableLambda(retrieve_and_format)
            | prompt
            | self._llm
            | StrOutputParser()
        )

        # 包裹多轮历史
        return RunnableWithMessageHistory(
            base_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _gen_marketing_tip(self, game_label: str, zh_label: str) -> str | None:
        trigger = MARKETING_TRIGGERS.get(game_label, "")
        promos  = "\n".join(f"- {p['name']}: {p['desc']}" for p in ACTIVE_PROMOTIONS)

        prompt = ChatPromptTemplate.from_messages([
            ("human",
             f"Customer intent: {zh_label}\nTrigger reason: {trigger}\n\n"
             f"Available promotions:\n{promos}\n\n"
             "Suggest ONE relevant promotion in 1 friendly sentence. "
             "Reply 'NO_SUGGESTION' if none fits."),
        ])
        chain  = prompt | self._llm | StrOutputParser()
        result = chain.invoke({}).strip()
        return None if result == "NO_SUGGESTION" else result

    def _escalation_reply(self) -> str:
        return (
            "I completely understand your concern! "
            "I'm connecting you with one of our specialist agents right now — "
            "they'll be with you shortly! 🎮"
        )