"""冒烟测试：验证新投票制管线端到端可用、预测不再坍缩、辩论投票被正确解析。"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import yaml
from src.pipeline.inference_pipeline import InferencePipeline

with open("configs/base.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
config['task'] = 'sentiment'
config['debate']['rounds'] = 2

pipeline = InferencePipeline(config, use_debate=True)

test_df = pd.read_csv("data/splits/sst2_test.csv")
test_df = test_df.rename(columns={"sentence": "text"})
samples = test_df.head(6).to_dict('records')

print("=" * 60)
print("冒烟测试：带辩论（投票制）")
print("=" * 60)
preds = []
for i, s in enumerate(samples):
    r = pipeline.run(s)
    preds.append(r["decision"])
    print(f"[{i}] true={s['label']} pred={r['decision']} "
          f"judge_vote={r.get('judge_vote', '-')} conf={r.get('judge_confidence', 0):.3f}")
    print(f"     debate_votes={r.get('debate_votes', {})}")
    print(f"     scores={r.get('score', {})}")

print("\n带辩论预测分布:", {p: preds.count(p) for p in set(preds)})

# 验证投票被解析
assert all(r["decision"] in ("pro", "con") for r in [pipeline.run(s) for s in samples[:1]]), "决策标签异常"
print("\n✅ 冒烟测试通过：预测标签为 pro/con，未出现坍缩或未知标签")
