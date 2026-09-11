"""纯 CPU 单元测试：验证票数门槛、基线保护、路由接线、门控前置的语义正确性。

不加载任何真实模型（GPU 正被 n=784 全量实验占用）。
关键设计：不覆写 JudgeAgent._classify，而是注入假 tokenizer/model，
让【真实的 _classify 代码路径（含缓存逻辑）】完整执行，否则缓存无法被验证。

覆盖的关键不变量：
  I1 默认配置（vote_threshold=2, gate=None, num_shots=0, retriever=None）
     行为必须与历史完全一致 —— 否则正在跑的全量实验结果不可比
  I2 基线保护：make_decision(debate_votes=None) 不得套用票数门槛
  I3 门控前置：被门控跳过的样本不得触发检索与辩论（省算力的前提）
  I4 路由接线：confidence_based 下低置信样本必须走 few_shot 且示例真传给辩论者
  I5 缓存：同一文本经门控+路由+计票多次访问，底层前向只发生 1 次
"""
import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from types import SimpleNamespace

from src.agents.judge_agent import JudgeAgent
from src.pipeline.inference_pipeline import InferencePipeline
from src.router.sample_aware_router import SampleAwareRouter

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "✓ PASS" if cond else "✗ FAIL"
    print(f"  [{mark}] {name}" + (f"\n          实际: {detail}" if not cond else ""))


# ══════════════════════════════════════════════════════════
# 假 tokenizer / model：让真实 JudgeAgent._classify 得以执行
# ══════════════════════════════════════════════════════════
class FakeTokenizer:
    def __init__(self, table):
        self.table = table
        self.last_text = None

    def __call__(self, text, return_tensors=None, truncation=True, max_length=128):
        self.last_text = text
        return {"input_ids": torch.tensor([[1, 2, 3]])}


class FakeModel:
    """按 text 查表产出 logits；统计真实前向次数以验证缓存。

    支持二分类与多分类：label2id 把标签名映射到 logit 位置，
    旧版只认 "pro"/其他 二分，导致多分类用例的 "tech" 落到 index 0（world）。
    """

    def __init__(self, tok, table, label2id):
        self.tok = tok
        self.table = table
        self.label2id = dict(label2id)
        self.n = len(label2id)
        self.forward_count = 0

    def __call__(self, **inputs):
        self.forward_count += 1
        label, conf = self.table.get(self.tok.last_text, (None, 0.5))
        conf = min(max(conf, 1e-6), 1 - 1e-6)
        idx = self.label2id.get(label)
        if idx is None:                      # 未登记标签 → 退回首个类别
            idx = 0
        # 其余类别均分剩余概率，保证目标类别的 softmax 恰为 conf
        rest = (1.0 - conf) / max(self.n - 1, 1)
        probs = [rest] * self.n
        probs[idx] = conf
        logits = torch.tensor([[math.log(p) for p in probs]])
        return SimpleNamespace(logits=logits)

    def to(self, device):
        return self

    def eval(self):
        return self


def make_judge(decision_map, gate_threshold=None, vote_threshold=2,
               judge_weight=1.0, debater_weight=1.0, mode="binary"):
    """构造 JudgeAgent 但绕过 __init__ 的权重下载，注入假 tokenizer/model。"""
    j = JudgeAgent.__new__(JudgeAgent)
    j.model_name = "stub"
    j.device = "cpu"
    j.max_length = 128
    j.judge_weight = judge_weight
    j.debater_weight = debater_weight
    j.gate_threshold = gate_threshold
    j.vote_threshold = int(vote_threshold)
    j._classify_cache = {}
    j.mode = mode
    j.id_to_label = ({0: "con", 1: "pro"} if mode == "binary"
                     else {0: "world", 1: "sports", 2: "business", 3: "tech"})
    j.tokenizer = FakeTokenizer(decision_map)
    label2id = {v: k for k, v in j.id_to_label.items()}
    j.model = FakeModel(j.tokenizer, decision_map, label2id)
    return j


class StubDebater:
    """可控投票的辩论者；记录收到的 examples 以验证路由真的传了示例。"""

    def __init__(self, pro_vote, con_vote, name="pro"):
        self.name = name
        self.vote = pro_vote if name == "pro" else con_vote
        self.received_examples = []
        self.calls = 0

    def reset(self):
        self.received_examples = []
        self.calls = 0

    def analyze(self, sample, examples=None):
        self.calls += 1
        self.received_examples.append(examples)
        return self.vote, f"arg {self.name}: FINAL ANSWER: {self.vote}"

    def defend_judgment(self, sample, own_vote, opponent_vote,
                        context="", examples=None):
        self.calls += 1
        self.received_examples.append(examples)
        return f"defend {own_vote}. FINAL ANSWER: {self.vote}"


class StubRetriever:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls = []

    def retrieve(self, text, k=3):
        self.calls.append({"text": text, "k": k})
        return [{"text": f"ex{i}", "label": i % 2, "score": 0.9 - i * 0.1}
                for i in range(k)]


def make_pipeline(judge, pro_vote="pro", con_vote="con", rounds=1,
                  retriever=None, num_shots=0, router_cfg=None, use_debate=True):
    cfg = {
        "task": "irony",
        "router": router_cfg or {"mode": "disabled"},
        "debate": {"rounds": rounds, "neutral": True},
    }
    p = InferencePipeline.__new__(InferencePipeline)   # 跳过 __init__ 的模型加载
    p.config = cfg
    p.router = SampleAwareRouter(cfg["router"])
    p.use_debate = use_debate
    p.task = "irony"
    p.retriever = retriever
    p.num_shots = num_shots
    p.pro_debater = StubDebater(pro_vote, con_vote, "pro")
    p.con_debater = StubDebater(pro_vote, con_vote, "con")
    p.judge = judge
    return p


SAMPLES = {
    "hi_conf_pro": ("pro", 0.97),    # 高置信，判反讽
    "hi_conf_con": ("con", 0.95),    # 高置信，判非反讽
    "lo_conf_pro": ("pro", 0.55),    # 低置信，判反讽
    "lo_conf_con": ("con", 0.58),    # 低置信，判非反讽
}
CONF_ROUTER = {"mode": "confidence_based", "confidence_threshold": 0.90,
               "few_shot_k": 3}


print("=" * 96)
print("I0 假模型自校验（确保 stub 本身可信）")
print("=" * 96)
j0 = make_judge(SAMPLES)
r0 = j0._classify("hi_conf_pro")
check("FakeModel 能复现预设判决 pro", r0["pred_label"] == "pro", str(r0))
check("FakeModel 能复现预设置信度 0.97", abs(r0["confidence"] - 0.97) < 1e-4,
      str(r0["confidence"]))
r0b = j0._classify("hi_conf_con")
check("FakeModel 能复现预设判决 con", r0b["pred_label"] == "con", str(r0b))
check("FakeModel 能复现预设置信度 0.95", abs(r0b["confidence"] - 0.95) < 1e-4,
      str(r0b["confidence"]))

print("\n" + "=" * 96)
print("I1 默认配置行为不变（与历史实验可比的前提）")
print("=" * 96)
j = make_judge(SAMPLES)
p = make_pipeline(j, retriever=None, num_shots=0, router_cfg={"mode": "disabled"})
r = p.run({"text": "lo_conf_pro", "label": 1})
check("默认门控关闭 → gated=False", r["gated"] is False, str(r.get("gated")))
check("默认零样本 → strategy=zero_shot", r["strategy"] == "zero_shot", r["strategy"])
check("默认无检索 → num_examples=0", r["num_examples"] == 0, str(r["num_examples"]))
check("辩论被调用（analyze+defend 各 1 次 = 2）",
      p.pro_debater.calls == 2, str(p.pro_debater.calls))
check("无检索器 → route.source=no_retriever（路由无从检索，沿用历史策略）",
      r["route"]["source"] == "no_retriever", str(r["route"]))
check("多数票 ≥2：judge=pro,正方=pro,反方=con → 2:1 → pro",
      r["decision"] == "pro", r["decision"])
check("vote_threshold 默认=2", r["vote_threshold"] == 2, str(r["vote_threshold"]))
check("pro_score=2.0", abs(r["pro_score"] - 2.0) < 1e-9, str(r["pro_score"]))
check("required_pro_score=2.0", abs(r["required_pro_score"] - 2.0) < 1e-9,
      str(r["required_pro_score"]))
check("★ 初始票被记录（归因偏置来源所需）",
      r["initial_votes"] == {"pro": "pro", "con": "con"}, str(r["initial_votes"]))

print("\n" + "=" * 96)
print("I2 基线保护（debate_votes=None 不得套用票数门槛）")
print("=" * 96)
j2 = make_judge(SAMPLES, vote_threshold=3)
base_pro = j2.make_decision({"text": "hi_conf_pro"}, debate_votes=None)
check("★ 门槛=3 时基线仍为 pro（核心保护，否则对照失效）",
      base_pro["decision"] == "pro", str(base_pro["decision"]))
check("基线·标记 baseline_no_debate=True",
      base_pro.get("baseline_no_debate") is True, str(base_pro.get("baseline_no_debate")))
check("基线·pro_score=None（未进入计票）",
      base_pro["pro_score"] is None, str(base_pro["pro_score"]))
base_empty = j2.make_decision({"text": "hi_conf_pro"}, debate_votes={})
check("基线·debate_votes={} → 仍走基线保护",
      base_empty["decision"] == "pro", str(base_empty["decision"]))
base_none = j2.make_decision({"text": "hi_conf_pro"},
                             debate_votes={"pro": None, "con": None})
check("基线·票全为 None → 仍走基线保护",
      base_none["decision"] == "pro", str(base_none["decision"]))
j2b = make_judge(SAMPLES, vote_threshold=2)
check("门槛=2 时基线也为 pro（历史行为一致）",
      j2b.make_decision({"text": "hi_conf_pro"}, debate_votes=None)["decision"] == "pro",
      "不一致")

print("\n" + "=" * 96)
print("I2b 票数门槛在真实辩论下的语义（≥2 vs ≥3）")
print("=" * 96)
# judge=con, 双方=pro → 2 票 pro
j3 = make_judge({"s": ("con", 0.6)}, vote_threshold=2)
p2 = make_pipeline(j3, pro_vote="pro", con_vote="pro")
r2 = p2.run({"text": "s", "label": 1})
check("≥2票门槛：judge=con + 双方=pro → 2票 → pro",
      r2["decision"] == "pro", f"decision={r2['decision']} score={r2['pro_score']}")
check("≥2票门槛：pro_score=2.0", abs(r2["pro_score"] - 2.0) < 1e-9, str(r2["pro_score"]))

j3b = make_judge({"s": ("con", 0.6)}, vote_threshold=3)
p3 = make_pipeline(j3b, pro_vote="pro", con_vote="pro")
r3 = p3.run({"text": "s", "label": 1})
check("★ ≥3票门槛：同样 2 票 → con（全票制生效，偏置被抑制）",
      r3["decision"] == "con", f"decision={r3['decision']} score={r3['pro_score']}")
check("≥3票门槛：required=3.0", abs(r3["required_pro_score"] - 3.0) < 1e-9,
      str(r3["required_pro_score"]))

j4 = make_judge({"s": ("pro", 0.6)}, vote_threshold=3)
p4 = make_pipeline(j4, pro_vote="pro", con_vote="pro")
r4 = p4.run({"text": "s", "label": 1})
check("≥3票门槛：三方全 pro → 3票 → pro",
      r4["decision"] == "pro", f"decision={r4['decision']} score={r4['pro_score']}")

j5 = make_judge({"s": ("con", 0.6)}, vote_threshold=3)
p5 = make_pipeline(j5, pro_vote="pro", con_vote="con")
r5 = p5.run({"text": "s", "label": 1})
check("≥3票：judge=con,正方=pro,反方=con → 1票 → con",
      r5["decision"] == "con", f"decision={r5['decision']} score={r5['pro_score']}")

# 不等权下的按比例缩放：三方全投 pro 才算满 3 票
j5b = make_judge({"s": ("pro", 0.6)}, vote_threshold=3,
                 judge_weight=1.0, debater_weight=1.0)
p5b = make_pipeline(j5b, pro_vote="pro", con_vote="pro")
r5b = p5b.run({"text": "s", "label": 1})
check("≥3票：judge=pro + 双方=pro → 3票 ≥ required 3.0 → pro",
      r5b["decision"] == "pro", f"score={r5b['pro_score']} req={r5b['required_pro_score']}")
check("≥3票：pro_score 恰为 3.0", abs(r5b["pro_score"] - 3.0) < 1e-9,
      str(r5b["pro_score"]))

# 权重不等时门槛按比例缩放：judge_weight=2, debater_weight=1
#   total_possible = 2 + 2*1 = 4, ratio = 3/3 = 1.0 → required = 4.0
j5c = make_judge({"s": ("pro", 0.6)}, vote_threshold=3,
                 judge_weight=2.0, debater_weight=1.0)
p5c = make_pipeline(j5c, pro_vote="pro", con_vote="pro")
r5c = p5c.run({"text": "s", "label": 1})
check("不等权：required 按 total_possible 缩放为 4.0",
      abs(r5c["required_pro_score"] - 4.0) < 1e-9, str(r5c["required_pro_score"]))
check("不等权：judge(2)+双方(2)=4.0 ≥ 4.0 → pro",
      r5c["decision"] == "pro", f"score={r5c['pro_score']}")

# 缺票时的缩放：只有正方有效票 → total_possible = 1+1 = 2 → required = 2.0
j5d = make_judge({"s": ("con", 0.6)}, vote_threshold=3)
p5d = make_pipeline(j5d, pro_vote="pro", con_vote=None)
# 让反方返回 None（解析失败），模拟缺票
p5d.con_debater.vote = None
r5d = p5d.run({"text": "s", "label": 1})
check("缺票（反方 None）→ required 按 total_possible=2 缩放",
      abs(r5d["required_pro_score"] - 2.0) < 1e-9,
      str(r5d["required_pro_score"]))

print("\n" + "=" * 96)
print("I3 门控前置（被跳过的样本不得触发检索与辩论）")
print("=" * 96)
ret = StubRetriever()
j6 = make_judge(SAMPLES, gate_threshold=0.90, vote_threshold=2)   # ★ 门控设在 judge 上
p6 = make_pipeline(j6, retriever=ret, num_shots=0, router_cfg=CONF_ROUTER)

p6.pro_debater.reset(); p6.con_debater.reset(); ret.reset()
r_hi = p6.run({"text": "hi_conf_pro", "label": 1})     # conf=0.97 ≥ 0.90 → 跳过
check("高置信样本 gated=True", r_hi["gated"] is True, str(r_hi.get("gated")))
check("★ 被门控跳过 → 检索调用数为 0（真实省算力）",
      len(ret.calls) == 0, f"检索被调用 {len(ret.calls)} 次")
check("被门控跳过 → 辩论者未被调用",
      p6.pro_debater.calls == 0, str(p6.pro_debater.calls))
check("被门控跳过 → decision=裁决者判决 pro",
      r_hi["decision"] == "pro", r_hi["decision"])
check("被门控跳过 → route.source=gated_skip",
      r_hi["route"]["source"] == "gated_skip", str(r_hi["route"]))
check("被门控跳过 → strategy=zero_shot",
      r_hi["strategy"] == "zero_shot", r_hi["strategy"])
check("被门控跳过 → num_examples=0", r_hi["num_examples"] == 0,
      str(r_hi["num_examples"]))

p6.pro_debater.reset(); p6.con_debater.reset(); ret.reset()
r_lo = p6.run({"text": "lo_conf_con", "label": 0})     # conf=0.58 < 0.90 → 进辩论
check("低置信样本 gated=False", r_lo["gated"] is False, str(r_lo.get("gated")))
check("★ 低置信 + confidence_based → few_shot（3_shot）",
      r_lo["strategy"] == "3_shot", r_lo["strategy"])
check("低置信 → 检索被调用 1 次", len(ret.calls) == 1, str(len(ret.calls)))
check("低置信 → 辩论者被调用（2 次）", p6.pro_debater.calls == 2,
      str(p6.pro_debater.calls))
check("★ 示例真的传给了辩论者（路由不是空转）",
      all(ex is not None and len(ex) == 3 for ex in p6.pro_debater.received_examples),
      str(p6.pro_debater.received_examples))
check("反方也收到了示例（双方对称）",
      all(ex is not None and len(ex) == 3 for ex in p6.con_debater.received_examples),
      str(p6.con_debater.received_examples))
check("检索 k 与 router.few_shot_k 一致",
      ret.calls and ret.calls[0]["k"] == 3, str(ret.calls))

# 边界：conf 恰等于阈值 → 跳过（门控用 confidence >= threshold）。
# 注意不能用字面量 0.90：FakeModel 经 log→softmax→exp 往返会有 1e-8 级误差，
# 故取 _classify 的【实际返回值】作为阈值，精确检验 >= 语义。
j6c = make_judge({"edge": ("pro", 0.90)})
actual_conf = j6c._classify("edge")["confidence"]
j6c.gate_threshold = actual_conf
p6c = make_pipeline(j6c, retriever=None, num_shots=0,
                    router_cfg={"mode": "disabled"})
r_edge = p6c.run({"text": "edge", "label": 1})
check("边界：conf == gate_threshold → 被跳过（gated=True）",
      r_edge["gated"] is True,
      f"gated={r_edge.get('gated')} conf={actual_conf!r}")
check("边界：跳过后 decision=裁决者判决", r_edge["decision"] == "pro",
      r_edge["decision"])

# 阈值略高于 conf → 不跳过
j6d = make_judge({"edge": ("pro", 0.90)})
j6d.gate_threshold = j6d._classify("edge")["confidence"] + 1e-6
p6d = make_pipeline(j6d, retriever=None, num_shots=0,
                    router_cfg={"mode": "disabled"})
r_edge2 = p6d.run({"text": "edge", "label": 1})
check("边界：conf 略低于阈值 → 不跳过（gated=False）",
      r_edge2["gated"] is False, f"gated={r_edge2.get('gated')}")

# 阈值略低于 conf → 跳过
j6e = make_judge({"edge": ("pro", 0.90)})
j6e.gate_threshold = j6e._classify("edge")["confidence"] - 1e-6
p6e = make_pipeline(j6e, retriever=None, num_shots=0,
                    router_cfg={"mode": "disabled"})
r_edge3 = p6e.run({"text": "edge", "label": 1})
check("边界：conf 略高于阈值 → 跳过（gated=True）",
      r_edge3["gated"] is True, f"gated={r_edge3.get('gated')}")

print("\n" + "=" * 96)
print("I4 路由模式对照")
print("=" * 96)
r_len = SampleAwareRouter({"mode": "length_based", "length_threshold": 100})
check("length_based：20 词 → zero_shot（复现死分支）",
      r_len.route({"text": "word " * 20}) == "zero_shot", r_len.last_decision["strategy"])
check("length_based：150 词 → few_shot（逻辑正确，只是阈值不适用短文本）",
      r_len.route({"text": "word " * 150}) == "few_shot", r_len.last_decision["strategy"])

r_conf = SampleAwareRouter(CONF_ROUTER)
check("confidence_based：conf=0.55 → few_shot",
      r_conf.route({"text": "t"}, confidence=0.55) == "few_shot",
      r_conf.last_decision["strategy"])
check("confidence_based：conf=0.97 → zero_shot",
      r_conf.route({"text": "t"}, confidence=0.97) == "zero_shot",
      r_conf.last_decision["strategy"])
check("confidence_based：未传置信度 → 回退长度判定（不静默全 zero_shot）",
      r_conf.route({"text": "word " * 20}) == "zero_shot", r_conf.last_decision["mode_used"])
check("confidence_based：回退时标注 fallback",
      "fallback" in r_conf.last_decision["mode_used"], r_conf.last_decision["mode_used"])
check("边界：conf == 阈值 → zero_shot（严格小于才 few_shot）",
      r_conf.route({"text": "t"}, confidence=0.90) == "zero_shot",
      r_conf.last_decision["strategy"])

r_d = SampleAwareRouter({"mode": "disabled"})
check("disabled：恒 zero_shot",
      r_d.route({"text": "t"}, confidence=0.01) == "zero_shot",
      r_d.last_decision["strategy"])

ret2 = StubRetriever()
j7 = make_judge(SAMPLES)
p7 = make_pipeline(j7, retriever=ret2, num_shots=5, router_cfg=CONF_ROUTER)
r7 = p7.run({"text": "lo_conf_pro", "label": 1})
check("显式 num_shots=5 → strategy=5_shot（路由不干预）",
      r7["strategy"] == "5_shot", r7["strategy"])
check("显式 num_shots=5 → 检索 k=5", ret2.calls[0]["k"] == 5, str(ret2.calls[0]["k"]))
check("显式 num_shots → route.source=explicit_num_shots",
      r7["route"]["source"] == "explicit_num_shots", str(r7["route"]))

j8 = make_judge(SAMPLES)
p8 = make_pipeline(j8, retriever=None, num_shots=3, router_cfg=CONF_ROUTER)
r8 = p8.run({"text": "lo_conf_pro", "label": 1})
check("无检索器 → route.source=no_retriever（不报错）",
      r8["route"]["source"] == "no_retriever", str(r8["route"]))
check("无检索器 + num_shots=3 → strategy 仍标注 3_shot",
      r8["strategy"] == "3_shot", r8["strategy"])
check("无检索器 → num_examples=0", r8["num_examples"] == 0, str(r8["num_examples"]))

print("\n" + "=" * 96)
print("I5 分类缓存（真实 _classify 路径，验证底层前向只发生 1 次）")
print("=" * 96)
j9 = make_judge(SAMPLES, gate_threshold=0.90, vote_threshold=3)
p9 = make_pipeline(j9, retriever=StubRetriever(), num_shots=0, router_cfg=CONF_ROUTER)
fwd_before = j9.model.forward_count
r9 = p9.run({"text": "lo_conf_pro", "label": 1})
fwd_after = j9.model.forward_count
check("★ 单条样本经 门控+路由+计票 → 底层前向只 1 次（缓存生效）",
      fwd_after - fwd_before == 1, f"实际前向 {fwd_after - fwd_before} 次")
c1 = j9._classify("lo_conf_pro")
c2 = j9._classify("lo_conf_pro")
check("缓存命中 → 结果完全一致", c1 == c2, f"{c1} vs {c2}")
fwd_n = j9.model.forward_count
j9._classify("lo_conf_pro")
check("缓存命中 → 底层前向未增加", j9.model.forward_count == fwd_n,
      str(j9.model.forward_count))
check("门控与计票使用同一置信度",
      abs(r9["judge_confidence"] - 0.55) < 1e-4, str(r9["judge_confidence"]))
check("不同文本各自缓存（不串味）",
      j9._classify("hi_conf_pro")["pred_label"] == "pro"
      and j9._classify("lo_conf_pro")["pred_label"] == "pro"
      and abs(j9._classify("hi_conf_pro")["confidence"]
              - j9._classify("lo_conf_pro")["confidence"]) > 0.3,
      "缓存键冲突")

print("\n" + "=" * 96)
print("I6 无辩论路径（use_debate=False）")
print("=" * 96)
j10 = make_judge(SAMPLES, gate_threshold=0.90, vote_threshold=3)
ret10 = StubRetriever()
p10 = make_pipeline(j10, use_debate=False, retriever=ret10, num_shots=0,
                    router_cfg=CONF_ROUTER)
r10 = p10.run({"text": "hi_conf_pro", "label": 1})
check("无辩论 + 高置信 → gated=True（门控前置生效）",
      r10["gated"] is True, str(r10["gated"]))
check("无辩论 + 高置信 → decision=pro（未被门槛压低）",
      r10["decision"] == "pro", r10["decision"])
p10.pro_debater.reset(); ret10.reset()
r10b = p10.run({"text": "lo_conf_con", "label": 0})
check("无辩论 + 低置信 → gated=False", r10b["gated"] is False, str(r10b["gated"]))
check("无辩论 + 低置信 → 路由 few_shot 生效",
      r10b["strategy"] == "3_shot", r10b["strategy"])
check("无辩论 → 仍返回 route 字段", "route" in r10b, str(list(r10b.keys())))
check("无辩论 → decision=裁决者判决 con（基线保护）",
      r10b["decision"] == "con", r10b["decision"])

print("\n" + "=" * 96)
print("I7 多分类不受票数门槛影响")
print("=" * 96)
jm = make_judge({"n": ("tech", 0.8)}, vote_threshold=3, mode="multiclass")
dm = jm.make_decision({"text": "n", "label": 3},
                      debate_votes={"pro": "tech", "con": "tech"})
check("多分类 → 得票最高者胜出（不走票数门槛）",
      dm["decision"] == "tech", dm["decision"])
check("多分类 → pro_score=None（门槛未介入）",
      dm["pro_score"] is None, str(dm["pro_score"]))
dm2 = jm.make_decision({"text": "n", "label": 3},
                       debate_votes={"pro": "sports", "con": "business"})
check("多分类·三方各异 → 平票由裁决者打破",
      dm2["decision"] == "tech", dm2["decision"])

print("\n" + "=" * 96)
print("I8 预设立场分支（neutral=False）的初始票记录")
print("=" * 96)
j11 = make_judge(SAMPLES)
p11 = make_pipeline(j11, pro_vote="pro", con_vote="con", retriever=None, num_shots=0,
                    router_cfg={"mode": "disabled"})
p11.config["debate"]["neutral"] = False
p11.config["debate"]["rounds"] = 2
# 预设立场分支调用 generate_argument，StubDebater 需提供该方法
p11.pro_debater.generate_argument = lambda s, ctx="", examples=None: (
    p11.pro_debater.received_examples.append(examples) or
    f"arg. FINAL ANSWER: {p11.pro_debater.vote}")
p11.con_debater.generate_argument = lambda s, ctx="", examples=None: (
    p11.con_debater.received_examples.append(examples) or
    f"arg. FINAL ANSWER: {p11.con_debater.vote}")
r11 = p11.run({"text": "lo_conf_pro", "label": 1})
check("预设立场分支 → initial_votes 被记录（首轮票）",
      r11["initial_votes"] == {"pro": "pro", "con": "con"}, str(r11["initial_votes"]))
check("预设立场分支 → debate_votes 为末轮票",
      r11["debate_votes"] == {"pro": "pro", "con": "con"}, str(r11["debate_votes"]))
check("预设立场分支 → 2 轮共 4 次生成（双方各 2）",
      len(p11.pro_debater.received_examples) == 2,
      str(len(p11.pro_debater.received_examples)))

print("\n" + "=" * 96)
print(f"测试结果: {len(PASS)} 通过 / {len(FAIL)} 失败")
print("=" * 96)
if FAIL:
    print("\n失败项:")
    for f in FAIL:
        print(f"  ✗ {f}")
    sys.exit(1)
print("""
★ 全部通过。核心结论：
  • 票数门槛 ≥3（全票制）语义正确，能抑制 pro 偏置
  • 基线保护生效：debate_votes 为空/全 None 时不套门槛，历史对照不被破坏
  • 门控前置生效：高置信样本不触发检索与辩论（真实省算力）
  • 路由接线生效：confidence_based 下低置信样本走 few_shot 且示例真传给双方
  • 缓存生效：门控+路由+计票共用 1 次底层前向
  • 默认配置行为与历史一致 → 正在跑的 n=784 实验结果仍可比
  • 初始票已记录 → 今后可归因偏置来自视角提示词还是质询轮漂移
""")
