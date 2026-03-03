"""
uplift_estimate.py
------------------
自动化效率估算：对比人工客服 vs AI Agent的成本与效率。
基于行业数据和项目评估结果，输出量化的效率提升报告。

运行：
    python scripts/uplift_estimate.py

输出：
    data/uplift_report.json
"""

import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("./data")
REPORT_PATH = DATA_DIR / "uplift_report.json"

# ══════════════════════════════════════════════════════════
# 基准假设（基于行业数据）
# ══════════════════════════════════════════════════════════
ASSUMPTIONS = {
    # 人工客服
    "human": {
        "avg_handle_time_min":    8.0,    # 平均每条工单处理时间（分钟）
        "hourly_cost_usd":        12.0,   # 人工客服时薪（美元，东南亚外包水平）
        "working_hours_per_day":  8.0,    # 每天工作小时数
        "tickets_per_agent_day":  60,     # 每名客服每天处理工单数
        "first_contact_resolution": 0.72, # 首次接触解决率
        "avg_response_time_min":  15.0,   # 平均响应时间（分钟）
    },
    # AI Agent（本项目）
    "ai": {
        "avg_handle_time_sec":    8.0,    # 平均响应时间（秒）
        "cost_per_1k_tokens_usd": 0.001,  # DeepSeek API成本
        "avg_tokens_per_ticket":  800,    # 每条工单平均token数
        "automation_rate":        0.80,   # 可自动化处理率（不需转人工）
        "first_contact_resolution": 0.85, # 首次接触解决率（基于评估结果）
        "availability":           "24/7", # 可用性
    },
    # 业务规模假设
    "scale": {
        "daily_tickets":          500,    # 每日工单量（中型游戏客服）
        "monthly_tickets":        15000,
        "annual_tickets":         180000,
    }
}


def calculate_human_cost(scale: dict, human: dict) -> dict:
    """计算纯人工客服的成本与效率。"""
    daily_tickets    = scale["daily_tickets"]
    tickets_per_day  = human["tickets_per_agent_day"]
    hourly_cost      = human["hourly_cost_usd"]
    handle_time_min  = human["avg_handle_time_min"]

    agents_needed      = daily_tickets / tickets_per_day
    daily_cost_usd     = agents_needed * hourly_cost * human["working_hours_per_day"]
    monthly_cost_usd   = daily_cost_usd * 30
    annual_cost_usd    = daily_cost_usd * 365
    cost_per_ticket    = daily_cost_usd / daily_tickets

    return {
        "agents_needed":       round(agents_needed, 1),
        "avg_handle_time_min": handle_time_min,
        "avg_response_time_min": human["avg_response_time_min"],
        "first_contact_resolution": human["first_contact_resolution"],
        "daily_cost_usd":      round(daily_cost_usd, 2),
        "monthly_cost_usd":    round(monthly_cost_usd, 2),
        "annual_cost_usd":     round(annual_cost_usd, 2),
        "cost_per_ticket_usd": round(cost_per_ticket, 4),
        "availability":        "Mon-Fri 9:00-18:00",
    }


def calculate_ai_cost(scale: dict, ai: dict, human: dict) -> dict:
    """计算AI Agent的成本与效率。"""
    daily_tickets     = scale["daily_tickets"]
    automation_rate   = ai["automation_rate"]
    cost_per_1k       = ai["cost_per_1k_tokens_usd"]
    avg_tokens        = ai["avg_tokens_per_ticket"]

    # AI处理的工单
    ai_handled        = daily_tickets * automation_rate
    # 转人工的工单
    human_handled     = daily_tickets * (1 - automation_rate)

    # AI API成本
    daily_api_cost    = daily_tickets * (avg_tokens / 1000) * cost_per_1k
    # 剩余人工成本
    reduced_agents    = human_handled / human["tickets_per_agent_day"]
    daily_human_cost  = reduced_agents * human["hourly_cost_usd"] * human["working_hours_per_day"]

    daily_total_cost  = daily_api_cost + daily_human_cost
    monthly_cost      = daily_total_cost * 30
    annual_cost       = daily_total_cost * 365
    cost_per_ticket   = daily_total_cost / daily_tickets

    return {
        "automation_rate":       automation_rate,
        "ai_handled_daily":      round(ai_handled),
        "human_handled_daily":   round(human_handled),
        "reduced_agents_needed": round(reduced_agents, 1),
        "avg_handle_time_sec":   ai["avg_handle_time_sec"],
        "avg_response_time_sec": ai["avg_handle_time_sec"],
        "first_contact_resolution": ai["first_contact_resolution"],
        "availability":          ai["availability"],
        "daily_api_cost_usd":    round(daily_api_cost, 2),
        "daily_human_cost_usd":  round(daily_human_cost, 2),
        "daily_total_cost_usd":  round(daily_total_cost, 2),
        "monthly_cost_usd":      round(monthly_cost, 2),
        "annual_cost_usd":       round(annual_cost, 2),
        "cost_per_ticket_usd":   round(cost_per_ticket, 4),
    }


def calculate_uplift(human: dict, ai: dict) -> dict:
    """计算效率提升指标。"""
    cost_reduction = (human["annual_cost_usd"] - ai["annual_cost_usd"]) / human["annual_cost_usd"]
    annual_savings = human["annual_cost_usd"] - ai["annual_cost_usd"]

    response_time_human_sec = human["avg_response_time_min"] * 60
    response_time_improvement = (response_time_human_sec - ai["avg_response_time_sec"]) / response_time_human_sec

    fcr_improvement = (ai["first_contact_resolution"] - human["first_contact_resolution"]) / human["first_contact_resolution"]

    return {
        "cost_reduction_rate":          round(cost_reduction, 4),
        "annual_savings_usd":           round(annual_savings, 2),
        "response_time_improvement":    round(response_time_improvement, 4),
        "response_time_human_min":      human["avg_response_time_min"],
        "response_time_ai_sec":         ai["avg_response_time_sec"],
        "fcr_improvement":              round(fcr_improvement, 4),
        "agents_reduced":               round(human["agents_needed"] - ai["reduced_agents_needed"], 1),
        "availability_upgrade":         f"{human['availability']} → {ai['availability']}",
        "automation_rate":              ai["automation_rate"],
    }


def load_eval_results() -> dict:
    """加载evaluate.py生成的评估报告，作为AI性能依据。"""
    eval_path = DATA_DIR / "eval_report.json"
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary", {})
    return {}


def main():
    print("=" * 55)
    print("  WhatsApp CRM — 自动化效率估算")
    print("=" * 55)

    human_cfg = ASSUMPTIONS["human"]
    ai_cfg    = ASSUMPTIONS["ai"]
    scale_cfg = ASSUMPTIONS["scale"]

    # 加载实测评估结果更新AI配置
    eval_results = load_eval_results()
    if eval_results:
        # 用实测的Intent准确率作为automation_rate的参考
        intent_acc = eval_results.get("intent_accuracy", 0.96)
        ai_cfg["automation_rate"] = round(min(intent_acc * 0.85, 0.90), 2)
        print(f"  ✅ 加载实测评估数据，automation_rate={ai_cfg['automation_rate']}")
    else:
        print("  ⚠️  未找到eval_report.json，使用默认假设值")

    human_cost = calculate_human_cost(scale_cfg, human_cfg)
    ai_cost    = calculate_ai_cost(scale_cfg, ai_cfg, human_cfg)
    uplift     = calculate_uplift(human_cost, ai_cost)

    # ── 打印结果 ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("  📊 成本对比（基于每日{:,}条工单）".format(scale_cfg["daily_tickets"]))
    print(f"{'─'*55}")
    print(f"  {'指标':<28} {'人工客服':>10} {'AI Agent':>10}")
    print(f"  {'─'*48}")
    print(f"  {'所需人力':<28} {human_cost['agents_needed']:>9.1f}人 {ai_cost['reduced_agents_needed']:>9.1f}人")
    print(f"  {'平均响应时间':<26} {human_cost['avg_response_time_min']:>8.0f}分钟 {ai_cost['avg_response_time_sec']:>8.0f}秒")
    print(f"  {'首次解决率(FCR)':<25} {human_cost['first_contact_resolution']:>9.0%} {ai_cost['first_contact_resolution']:>9.0%}")
    print(f"  {'每工单成本':<27} ${human_cost['cost_per_ticket_usd']:>8.4f} ${ai_cost['cost_per_ticket_usd']:>8.4f}")
    print(f"  {'月度总成本':<27} ${human_cost['monthly_cost_usd']:>8,.0f} ${ai_cost['monthly_cost_usd']:>8,.0f}")
    print(f"  {'年度总成本':<27} ${human_cost['annual_cost_usd']:>8,.0f} ${ai_cost['annual_cost_usd']:>8,.0f}")
    print(f"  {'可用性':<29} {human_cost['availability']:>10} {ai_cost['availability']:>10}")

    print(f"\n{'─'*55}")
    print("  🚀 效率提升汇总")
    print(f"{'─'*55}")
    print(f"  成本降低率:        {uplift['cost_reduction_rate']:.1%}")
    print(f"  年度节省:          ${uplift['annual_savings_usd']:,.0f}")
    print(f"  响应时间提升:      {uplift['response_time_improvement']:.1%}  ({uplift['response_time_human_min']:.0f}分钟 → {uplift['response_time_ai_sec']:.0f}秒)")
    print(f"  FCR提升:           {uplift['fcr_improvement']:+.1%}")
    print(f"  可释放人力:        {uplift['agents_reduced']:.1f}人")
    print(f"  自动化率:          {uplift['automation_rate']:.0%}")

    # ── 保存报告 ─────────────────────────────────────────
    report = {
        "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assumptions":   ASSUMPTIONS,
        "eval_results_used": eval_results,
        "human_baseline": human_cost,
        "ai_projection":  ai_cost,
        "uplift_summary": uplift,
        "scale":          scale_cfg,
    }

    DATA_DIR.mkdir(exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  📄 完整报告已保存至 {REPORT_PATH}")


if __name__ == "__main__":
    main()
