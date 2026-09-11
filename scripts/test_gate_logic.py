"""门控逻辑单元测试：不加载任何模型，仅验证 JudgeAgent 的门控判定与短路计票。

覆盖：
  1) 高置信 → 短路，忽略辩论投票，gated=True
  2) 低置信 → 进入投票聚合，gated=False，辩论可翻转裁决者
  3) 门控关闭（threshold=None）→ 行为与改动前一致
  4) should_debate 的判定方向与阈值边界
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.judge_agent import JudgeAgent


def make_judge(conf, label, gate):
    """构造不加载模型的裁决者，桩掉 _classify。"""
    j = JudgeAgent.__new__(JudgeAgent)
    j.judge_weight = 1.0
    j.debater_weight = 1.0
    j.gate_threshold = gate
    j._classify = lambda text: {
        "pred_id": 1 if label == "pro" else 0,
        "pred_label": label,
        "confidence": conf,
        "prob_vector": [1 - conf, conf] if label == "pro" else [conf, 1 - conf],
    }
    return j


S = {"text": "x"}

# ── 1) 高置信短路：裁决者判 con 且 conf=0.92 ≥ τ=0.6，辩论双方都投 pro 也应被忽略 ──
j = make_judge(conf=0.92, label="con", gate=0.6)
r = j.make_decision(S, debate_votes={"pro": "pro", "con": "pro"})
assert r["decision"] == "con", f"高置信未短路: {r['decision']}"
assert r["gated"] is True, "gated 标记错误"
assert r["debate_votes"] == {}, "短路时不应记录辩论投票"
print("✓ 场景1 高置信短路: decision=con, gated=True, 辩论投票被忽略")

# ── 2) 低置信辩论：conf=0.55 < τ=0.6，裁决者 con，两位辩论者投 pro → 多数表决应为 pro ──
j = make_judge(conf=0.55, label="con", gate=0.6)
r = j.make_decision(S, debate_votes={"pro": "pro", "con": "pro"})
assert r["decision"] == "pro", f"低置信未被辩论翻转: {r['decision']}"
assert r["gated"] is False, "gated 标记错误"
print("✓ 场景2 低置信辩论: decision=pro（辩论成功翻转裁决者）, gated=False")

# ── 3) 低置信但辩论分歧：裁决者 con + pro/con 各一票 → 平票由裁决者裁决 = con ──
j = make_judge(conf=0.52, label="con", gate=0.6)
r = j.make_decision(S, debate_votes={"pro": "pro", "con": "con"})
assert r["decision"] == "con", f"平票裁决错误: {r['decision']}"
print("✓ 场景3 低置信平票: decision=con（裁决者打破平局）")

# ── 4) 门控关闭：threshold=None 时高置信也走投票聚合 ──
j = make_judge(conf=0.99, label="con", gate=None)
r = j.make_decision(S, debate_votes={"pro": "pro", "con": "pro"})
assert r["decision"] == "pro", f"门控关闭时未走投票: {r['decision']}"
assert r["gated"] is False, "门控关闭时 gated 应为 False"
print("✓ 场景4 门控关闭: 高置信仍参与投票聚合（行为与改动前一致）")

# ── 5) 阈值边界：conf == τ 应判为"不需辩论"（>= 短路）──
j = make_judge(conf=0.6, label="pro", gate=0.6)
assert j.should_debate(S)["debate"] is False, "边界 conf==τ 应短路"
j = make_judge(conf=0.599, label="pro", gate=0.6)
assert j.should_debate(S)["debate"] is True, "conf<τ 应触发辩论"
print("✓ 场景5 阈值边界: conf==τ 短路，conf<τ 触发辩论")

# ── 6) should_debate 关闭门控时恒为 True ──
j = make_judge(conf=0.99, label="pro", gate=None)
assert j.should_debate(S)["debate"] is True, "门控关闭时应恒触发辩论"
print("✓ 场景6 门控关闭时 should_debate 恒为 True")

print("\n✅ 门控逻辑全部单元测试通过")
