"""
config.py
---------
统一配置管理，所有模块从这里读取配置。
"""
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, "")


# ── API Keys ─────────────────────────────────────────────
DASHSCOPE_API_KEY = _get_secret("DASHSCOPE_API_KEY")
DEEPSEEK_API_KEY  = _get_secret("DEEPSEEK_API_KEY")

# ── 模型配置 ─────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "deepseek-chat")
RERANK_MODEL    = os.getenv("RERANK_MODEL", "gte-rerank")   # 通义千问Rerank模型

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "data" / "chroma_db"))
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "game_crm_kb")

# ── 检索配置（Hybrid Search两阶段） ──────────────────────
# 第一阶段：粗排召回
DENSE_TOP_K  = int(os.getenv("DENSE_TOP_K",  "20"))   # Dense向量检索召回数
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "20"))   # BM25关键词检索召回数
RRF_K        = int(os.getenv("RRF_K", "60"))           # RRF融合参数（标准值60）
# 第二阶段：精排后返回数
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))    # Rerank后最终返回数

# ── 文本分割 ─────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
SEPARATORS    = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]

# ── 去重 ─────────────────────────────────────────────────
FINGERPRINT_FILE = str(_PROJECT_ROOT / "data" / "kb_fingerprints.txt")

# ── 对话历史 ─────────────────────────────────────────────
CHAT_HISTORY_DIR = os.getenv("CHAT_HISTORY_DIR", str(_PROJECT_ROOT / "data" / "chat_history"))

# ── Intent taxonomy：通用客服intent → 游戏CRM映射 ─────────
INTENT_TAXONOMY = {
    "create_account":        {"game_label": "account_register",  "zh": "账号注册"},
    "delete_account":        {"game_label": "account_delete",    "zh": "注销账号"},
    "edit_account":          {"game_label": "account_edit",      "zh": "账号信息修改"},
    "recover_password":      {"game_label": "account_recovery",  "zh": "账号找回"},
    "registration_problems": {"game_label": "account_issue",     "zh": "注册/登录问题"},
    "switch_account":        {"game_label": "account_switch",    "zh": "切换账号"},
    "payment_issue":         {"game_label": "payment_failed",    "zh": "充值/支付失败"},
    "check_payment_methods": {"game_label": "payment_methods",   "zh": "支付方式查询"},
    "get_refund":            {"game_label": "refund_request",    "zh": "退款申请"},
    "check_refund_policy":   {"game_label": "refund_policy",     "zh": "退款政策查询"},
    "cancel_order":          {"game_label": "purchase_cancel",   "zh": "取消购买"},
    "check_invoice":         {"game_label": "purchase_history",  "zh": "购买记录查询"},
    "track_order":           {"game_label": "item_delivery",     "zh": "道具到账查询"},
    "place_order":           {"game_label": "in_game_purchase",  "zh": "游戏内购咨询"},
    "complaint":             {"game_label": "complaint",         "zh": "投诉"},
    "contact_human_agent":   {"game_label": "escalate_to_human", "zh": "转人工客服"},
    "contact_customer_service": {"game_label": "general_inquiry","zh": "一般咨询"},
    "review":                {"game_label": "feedback",          "zh": "反馈/评价"},
    "newsletter_subscription": {"game_label": "event_subscribe", "zh": "活动订阅"},
    "delivery_options":      {"game_label": "gift_options",      "zh": "礼包选项"},
}

# ── 营销触发规则 ─────────────────────────────────────────
MARKETING_TRIGGERS = {
    "payment_failed":   "用户有充值意向但遇到问题，推荐其他支付方式或礼包",
    "payment_methods":  "用户了解支付方式，告知当前优惠活动",
    "refund_request":   "用户退款可能流失，推荐等值道具补偿",
    "in_game_purchase": "高意向用户，推荐当前性价比最高礼包",
    "event_subscribe":  "用户对活动感兴趣，推荐最新活动",
}

ACTIVE_PROMOTIONS = [
    {"name": "限时双倍经验", "desc": "本周末充值任意金额，经验值双倍，截止周日23:59"},
    {"name": "新手礼包特惠", "desc": "新用户专属，含1000金币+稀有角色碎片×5，限时24小时"},
    {"name": "月卡超值订阅", "desc": "30天仅需$4.99，每日额外奖励，今日购买赠7天"},
]
