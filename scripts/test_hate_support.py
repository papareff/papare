"""验证 hate 任务支持：投票解析、示例标签、对称视角提示词。

纯 CPU，不加载模型（GPU 正被 n=784 全量实验占用）。
这是链式队列第一步（hate 探针）的前置检查——若视角提示词不对称，
会重演 irony 任务的视角倒挂（反方投 pro 率反高 15.3pp）。

写成独立文件而非 PowerShell 内联 -c：内联多行字符串含单引号会被
PowerShell 提前截断，导致静默失败（本项目已踩坑两次）。
"""
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.debater_agent import (
    parse_vote, format_examples, HATE_LABEL_NAMES, DebaterAgent)

ok, bad = 0, 0


def chk(name, cond, detail=""):
    global ok, bad
    if cond:
        ok += 1
    else:
        bad += 1
        print(f"  [FAIL] {name}   {detail}")


print("=" * 96)
print("hate 任务支持验证")
print("=" * 96)

# ── 1. parse_vote：hate 使用 pro/con 词表 ──
print("\n【1】parse_vote 投票解析")
chk("FINAL ANSWER: pro → pro", parse_vote("blah FINAL ANSWER: pro", "hate") == "pro")
chk("FINAL ANSWER: con → con", parse_vote("blah FINAL ANSWER: con", "hate") == "con")
chk("FINAL ANSWER: hateful → pro",
    parse_vote("x FINAL ANSWER: hateful", "hate") == "pro")
chk("FINAL ANSWER: not hateful → con",
    parse_vote("x FINAL ANSWER: not hateful", "hate") == "con")
chk("回退：裸 pro 词", parse_vote("the tweet is hateful overall pro", "hate") == "pro")
chk("多次出现取最后一次",
    parse_vote("FINAL ANSWER: con ... reconsider ... FINAL ANSWER: pro", "hate") == "pro")

# ── 2. 不影响既有任务 ──
print("\n【2】既有任务未受影响（回归）")
chk("irony: ironic → pro", parse_vote("x FINAL ANSWER: ironic", "irony") == "pro")
chk("irony: not ironic → con",
    parse_vote("x FINAL ANSWER: not ironic", "irony") == "con")
chk("sentiment: positive → pro",
    parse_vote("x FINAL ANSWER: positive", "sentiment") == "pro")
chk("multiclass: sports → sports",
    parse_vote("FINAL ANSWER: sports", "agnews") == "sports")


class _S:
    task = "irony"; stance = "pro"


class _SC:
    task = "irony"; stance = "con"


chk("irony pro 视角仍为 IRONY DETECTION",
    "IRONY DETECTION" in DebaterAgent._lens(_S()))
chk("irony con 视角仍为 LITERAL READING",
    "LITERAL READING" in DebaterAgent._lens(_SC()))

# ── 3. format_examples：hate 标签展示 ──
print("\n【3】format_examples 示例标签")
ex = [{"text": "t1", "label": 1, "score": 0.9},
      {"text": "t2", "label": 0, "score": 0.8}]
s = format_examples(ex, "hate")
chk("hate 示例含 hateful / not hateful",
    "hateful" in s and "not hateful" in s, s)
chk("hate 示例不含多分类词表",
    "world" not in s and "sports" not in s and "tech" not in s, s)
chk("HATE_LABEL_NAMES 映射正确",
    HATE_LABEL_NAMES == {0: "not hateful", 1: "hateful"}, str(HATE_LABEL_NAMES))

# ── 4. _lens：hate 视角必须对称（核心，防重演 irony 倒挂）──
print("\n【4】_lens 视角对称性（防止重演 irony 视角倒挂）")


class _HP:
    task = "hate"; stance = "pro"


class _HC:
    task = "hate"; stance = "con"


lp = DebaterAgent._lens(_HP())
lc = DebaterAgent._lens(_HC())
chk("正方视角 = EXPLICIT HOSTILITY DETECTION", "EXPLICIT HOSTILITY" in lp, lp[:60])
chk("反方视角 = IMPLICIT HOSTILITY DETECTION", "IMPLICIT HOSTILITY" in lc, lc[:60])
chk("两方视角文本不同", lp != lc)


def n_cues(t):
    """统计具体可检验线索数。irony 的教训：正方 7 类具体证据 vs 反方 4 个抽象词
    → 反方无约束、回落任务先验、投 pro 率反高 15.3pp。故两方都需具体线索。"""
    pat = (r"slurs|dehumanizing|threats|violence|exclude|harm|insults|identity|"
           r"race|gender|religion|origin|orientation|"
           r"mocking|contemptuous|stereotyping|false praise|belittles|"
           r"rhetorical questions|inferiority|quoting|ridicule")
    return len(re.findall(pat, t, re.I))


def n_abstract(t):
    return len(re.findall(r"\bsimply\b|\bsincere\b|\bplain\b|\bstraightforward\b",
                          t, re.I))


cp, cc = n_cues(lp), n_cues(lc)
wp, wc = len(lp.split()), len(lc.split())
print(f"    正方: 具体线索 {cp} 类, 抽象词 {n_abstract(lp)} 个, {wp} 词")
print(f"    反方: 具体线索 {cc} 类, 抽象词 {n_abstract(lc)} 个, {wc} 词")
chk("两方都有具体可检验线索（≥4 类）", cp >= 4 and cc >= 4, f"pro={cp} con={cc}")
chk("两方线索数接近（差 ≤5）", abs(cp - cc) <= 5, f"pro={cp} con={cc}")
chk("两方词数接近（差 ≤15）", abs(wp - wc) <= 15, f"pro={wp} con={wc}")
chk("反方无抽象形容词堆砌（≤1 个）", n_abstract(lc) <= 1, str(n_abstract(lc)))
chk("反方无否定启动表述（hidden opposite）",
    "hidden opposite" not in lc.lower(), lc[:80])
chk("两方均以同一句式收尾（认知负担对称）",
    lp.strip().endswith("then give your verdict.")
    and lc.strip().endswith("then give your verdict."))

# ── 5. _classify_prompt：hate 走 pro/con 而非多分类 ──
print("\n【5】_classify_prompt 与 defend_judgment 分支")


class _StubAgent:
    task = "hate"
    stance = "pro"
    _lens = DebaterAgent._lens

    @staticmethod
    def generate(prompt, **kw):
        # defend_judgment 末尾会调用 self.generate；这里回显提示词以便断言其内容
        return prompt


cp_text = DebaterAgent._classify_prompt(_StubAgent(), {"text": "you are all idiots"})
chk("分类提示词含 hateful / not hateful 二选一",
    "'pro' if HATEFUL" in cp_text and "'con' if NOT HATEFUL" in cp_text,
    cp_text[-160:])
chk("分类提示词不含多分类类别词",
    "world" not in cp_text and "sports" not in cp_text and "business" not in cp_text)
chk("分类提示词含 EXPLICIT HOSTILITY 视角", "EXPLICIT HOSTILITY" in cp_text)

dj = DebaterAgent.defend_judgment(_StubAgent(), {"text": "x"}, "pro", "con",
                                  context="some reasoning")
chk("质询提示词含 HATEFUL 标签名", "HATEFUL" in dj, dj[:120])
chk("质询提示词不含 news category", "news category" not in dj)
chk("质询提示词含如实的关系描述（DISAGREED）", "DISAGREED" in dj)
chk("质询提示词含对方论证上下文", "some reasoning" in dj)
chk("质询提示词含 hate 视角（EXPLICIT HOSTILITY）", "EXPLICIT HOSTILITY" in dj)

print("\n" + "=" * 96)
print(f"结果: {ok} 通过 / {bad} 失败")
print("=" * 96)
if bad:
    print("\n失败项见上方 [FAIL] 标记。")
    sys.exit(1)
print("""
★ hate 任务支持完整，且视角提示词严格对称（两方均给具体可检验线索、
  词数接近、反方无抽象堆砌与否定启动）。链式队列的 hate 探针可以启动。

  标签语义: hate label=1 → hateful → pro 票 ; label=0 → not hateful → con 票
  正方视角: 显性敌意（脏词/非人化/暴力威胁/身份攻击）
  反方视角: 隐性敌意（嘲讽语气/刻板印象/虚假赞美/贬低性反问/引用以羞辱）
  → 两个视角都有独立发现能力，不构成 irony 式的「一方有约束一方无约束」
""")
