"""
safety.py
---------
安全与品牌合规层（输入拦截 + 输出合规检查）。
"""
import re
from dataclasses import dataclass, field

# ── 输入拦截规则 ─────────────────────────────────────────
INPUT_RULES = [
    (r"ignore (previous|all) instructions?",      "prompt_injection"),
    (r"you are now",                               "role_override"),
    (r"act as (a )?(different|unrestricted)",      "role_override"),
    (r"\b(hack|exploit|cheat engine|mod menu)\b", "cheating_tool"),
    (r"\b(kill|suicide|self.harm)\b",              "harmful_content"),
    (r"\b(credit card|ssn|social security)\b",    "pii_request"),
]

# ── 输出硬性规则（触发即拦截） ───────────────────────────
OUTPUT_HARD_RULES = [
    (r"\bas an AI\b",              "ai_disclosure"),
    (r"\bguarantee(d)?\b",        "false_guarantee"),
    (r"\$\d+(\.\d{1,2})?",        "exact_refund_amount"),
]

# ── 输出tone警告（标记不拦截） ───────────────────────────
OUTPUT_TONE_RULES = [
    (r"[A-Z]{5,}",    "warn:excessive_caps"),
    (r"!{3,}",        "warn:excessive_exclamation"),
]

INPUT_FALLBACK = (
    "Thank you for contacting us! For security reasons, we're unable to process "
    "this request through chat. Please visit our official support portal. 🙏"
)
OUTPUT_FALLBACK = (
    "Thank you for reaching out! Our team is reviewing your request and will "
    "respond as soon as possible. For urgent matters, contact our 24/7 support line. 🎮"
)


@dataclass
class SafetyResult:
    is_blocked:     bool
    flags:          list[str] = field(default_factory=list)
    fallback_reply: str = ""


class SafetyGuard:
    def __init__(self):
        self._input_rules       = [(re.compile(p, re.I), l) for p, l in INPUT_RULES]
        self._output_hard_rules = [(re.compile(p, re.I), l) for p, l in OUTPUT_HARD_RULES]
        self._output_tone_rules = [(re.compile(p, re.I), l) for p, l in OUTPUT_TONE_RULES]

    def check_input(self, text: str) -> SafetyResult:
        flags = [label for pat, label in self._input_rules if pat.search(text)]
        blocked = bool(flags)
        return SafetyResult(blocked, flags, INPUT_FALLBACK if blocked else "")

    def check_output(self, text: str) -> SafetyResult:
        hard_flags = [label for pat, label in self._output_hard_rules if pat.search(text)]
        tone_flags = [label for pat, label in self._output_tone_rules if pat.search(text)]
        flags      = hard_flags + tone_flags
        blocked    = bool(hard_flags)
        return SafetyResult(blocked, flags, OUTPUT_FALLBACK if blocked else "")
