"""单元测试：验证裁决者投票聚合逻辑（不加载模型）。"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.judge_agent import JudgeAgent

# 构造不加载模型的实例，只测试计票逻辑
j = JudgeAgent.__new__(JudgeAgent)
j.judge_weight = 1.0
j.debater_weight = 1.0

# 场景1：裁决者判 con，两位辩论者均投 pro → 多数表决应为 pro
j._classify = lambda text: {"pred_id": 0, "pred_label": "con", "confidence": 0.99, "prob_vector": [0.99, 0.01]}
r = j.make_decision({"text": "x"}, {"pro": "pro", "con": "pro"})
assert r["decision"] == "pro", f"场景1失败: {r['decision']}"
print("场景1通过: 裁决者con + 辩论者pro×2 →", r["decision"], "score:", r["score"])

# 场景2：裁决者判 pro，辩论者一正一反 → 平票(2:1:1)由裁决者裁决，应为 pro
j._classify = lambda text: {"pred_id": 1, "pred_label": "pro", "confidence": 0.8, "prob_vector": [0.2, 0.8]}
r = j.make_decision({"text": "x"}, {"pro": "pro", "con": "con"})
assert r["decision"] == "pro", f"场景2失败: {r['decision']}"
print("场景2通过: 裁决者pro + 辩论者pro/con →", r["decision"], "score:", r["score"])

# 场景3：无辩论（消融基线）→ 裁决者独立判决
j._classify = lambda text: {"pred_id": 0, "pred_label": "con", "confidence": 0.9, "prob_vector": [0.9, 0.1]}
r = j.make_decision({"text": "x"}, debate_votes=None)
assert r["decision"] == "con", f"场景3失败: {r['decision']}"
print("场景3通过: 无辩论 → 裁决者独立判决", r["decision"])

# 场景4：辩论者投票解析失败（None）→ 只有裁决者有效票
r = j.make_decision({"text": "x"}, {"pro": None, "con": None})
assert r["decision"] == "con", f"场景4失败: {r['decision']}"
print("场景4通过: 投票解析失败 → 裁决者判决", r["decision"])

print("\n✅ 计票逻辑全部单元测试通过")
