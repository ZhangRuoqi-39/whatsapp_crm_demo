"""
intent.py
"""
import os
import random
from dataclasses import dataclass

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import DEEPSEEK_API_KEY, CHAT_MODEL, INTENT_TAXONOMY

TRAIN_CSV = "./data/bitext_intents.csv"
N_SHOTS   = 2


@dataclass
class IntentResult:
    game_label: str
    zh_label:   str
    confidence: str
    reasoning:  str


class IntentClassifier:
    def __init__(self):
        self._llm = ChatOpenAI(
            model=CHAT_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            temperature=0.0,
        )
        self._examples     = self._load_examples()
        self._valid_labels = {v["game_label"] for v in INTENT_TAXONOMY.values()}
        self._system_prompt = self._build_system_prompt()
        self._parser = JsonOutputParser()

    def _load_examples(self) -> dict[str, list[str]]:
        if not os.path.exists(TRAIN_CSV):
            return {}
        df = pd.read_csv(TRAIN_CSV)
        examples = {}
        for intent in df["intent"].unique():
            subset = df[df["intent"] == intent]["user_message"].tolist()
            examples[intent] = random.sample(subset, min(N_SHOTS, len(subset)))
        return examples

    def _build_intent_block(self) -> str:
        lines = []
        for raw, info in INTENT_TAXONOMY.items():
            exs    = self._examples.get(raw, [])
            ex_str = " | ".join(f'"{e}"' for e in exs)
            lines.append(f"- {info['game_label']} ({info['zh']})\n  Examples: {ex_str}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        intent_block = self._build_intent_block()
        return (
            "You are an intent classifier for a live-service game WhatsApp CRM system.\n"
            "The customer message may be in Chinese or English.\n\n"
            "Classify the message into one of these intents:\n\n"
            + intent_block +
            "\n\nRULES:\n"
            "1. Return ONLY valid JSON, no extra text, no markdown.\n"
            "2. JSON keys: game_label, zh_label, confidence (high/medium/low), reasoning.\n"
            "3. Chinese hints: "
            "chong-zhi/payment/gems=payment_failed, "
            "tui-kuan/refund=refund_request, "
            "zhang-hao/account=account related, "
            "dao-ju/item=item_delivery\n\n"
            'Example: {"game_label":"payment_failed","zh_label":"充值/支付失败",'
            '"confidence":"high","reasoning":"User reports failed payment."}'
        )

    def classify(self, message: str) -> IntentResult:
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"Classify this customer message: {message}"),
        ]
        try:
            response = self._llm.invoke(messages)
            raw = self._parser.invoke(response)
        except Exception as e:
            print(f"Intent分类异常: {e}")
            raw = {}

        label = raw.get("game_label", "general_inquiry")
        if label not in self._valid_labels:
            label = "general_inquiry"

        zh = next(
            (v["zh"] for v in INTENT_TAXONOMY.values() if v["game_label"] == label),
            "一般咨询",
        )
        return IntentResult(
            game_label=label,
            zh_label=raw.get("zh_label", zh),
            confidence=raw.get("confidence", "low"),
            reasoning=raw.get("reasoning", ""),
        )