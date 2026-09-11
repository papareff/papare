"""困难任务候选数据集探测：为泛化性实验筛选「困难且与反讽同族」的任务。

筛选标准（针对 AG News 被否决的教训）：
  ① 必须困难——基线准确率显著低于饱和线（SST-2 裁决者 98% 属饱和，辩论无纠错空间）
     判据：人类一致率低、或 SOTA 明显低于 85%、或类别高度歧义
  ② 最好与反讽同族（语用/隐含意义），使「视角分化辩论」的机理可迁移
  ③ 二分类优先（当前框架的票数门槛、偏置分析都建立在二分类上）
  ④ 规模适中（测试集 ≤5000，避免 AG News 7600 条 × 24.45s = 50h 的成本）
  ⑤ 可合法获取、标注可靠（Misra 按新闻源标注导致模型靠文风作弊，是反面教材）

候选任务族：
  A. 带上下文的讽刺/反讽（contextual sarcasm）—— 与反讽最接近，且上下文引入新难度
  B. 隐含仇恨言论 / 隐性攻击（implicit hate）—— 表层无攻击词但实际攻击，
     与反讽的「字面≠意图」结构同构
  C. 蕴含/反语（sarcasm entailment）
  D.  offensiveness 细粒度（TRAC / OffensEval）
"""
import urllib.request
import json
import os

BASE = "https://hf-mirror.com"

# (数据集 ID, 任务族, 规模预期, 难度依据)
CANDIDATES = [
    # ── A. 带上下文的讽刺 ──
    ("siddharthyadev/sarcasm-2.0-reddit", "A 上下文讽刺", "4400",
     "Reddit 对话，需结合上文判断回复是否讽刺；上下文依赖=新难度"),
    ("CreativeLang/SARC_Sarcasm", "A 上下文讽刺", "3176188(分片)",
     "SARC 2.0 含 context 字段，规模最大但弱标注"),
    ("macabdul9/SarcasmDetection_Mustard", "A 上下文讽刺(多模态)", "3405",
     "MUSTARD 多模态讽刺，含视频；文本部分可用"),
    ("DynamicSuperb/SarcasmDetection_Mustard", "A 上下文讽刺(多模态)", "3405",
     "同上另一份镜像"),
    ("yl2342/friends_chandler_bing_sarcasm", "A 对话讽刺", "1222",
     "《老友记》对话讽刺，需对话上下文"),
    # ── B. 隐含仇恨 / 隐性攻击 ──
    ("michellemwange25/hate_speech_offensive", "B 隐含仇恨", "24783",
     "Davidson 经典集，含 offensive/hate/neither 三类，hate 类高度隐性"),
    ("SetFit/implicit_hate_v1", "B 隐含仇恨", "4979",
     "★ 专门标注【隐性】仇恨（implicit hate），表层无攻击词，与反讽同构"),
    ("SetFit/hate_speech18", "B 仇恨言论", "10568",
     "Reddit 仇恨言论，标注噪声较大"),
    ("dair-ai/emotion", "B 参照(情感)", "16000",
     "6 类情感，饱和任务，仅作对照不推荐"),
    ("SetFit/ tweet_eval_hate", "B 仇恨", "—", "TweetEval hate 子集"),
    ("tweet_eval", "B 仇恨(子集)", "hate 10240",
     "TweetEval 含 hate/offensive/irony/stance 多子集，★ irony 即当前任务"),
    # ── C/D. 攻击性细粒度 ──
    ("cardiffnlp/tweet_sentiment_multilingual", "D 参照", "—", "多语情感，饱和"),
    ("mteb/tweet-sentiment-extraction", "D 参照", "27481", "情感抽取，非分类"),
    # ── 补充：讽刺的学术标注集 ──
    ("Lots-of-LoRAs/task1489_sarcasmdetection_tweet_classification", "A 推特讽刺", "1200",
     "Natural Instructions 格式，规模小"),
    ("tasksource/figlang2020-sarcasm", "A FIGLANG", "3419",
     "FIGLANG 2020 共享任务，推特讽刺，标注质量高"),
    ("winduald/figlang2020-sarcasm", "A FIGLANG", "3419", "同上另一份"),
]


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def probe(ds_id):
    out = {"id": ds_id, "exists": False}
    try:
        info = json.loads(get(f"{BASE}/api/datasets/{ds_id}"))
        out["exists"] = True
        out["downloads"] = info.get("downloads", 0)
        out["likes"] = info.get("likes", 0)
        sib = info.get("siblings") or []
        out["files"] = [s.get("rfilename") for s in sib][:20]
        out["n_files"] = len(sib)
        # 标签体系（若 README 有 dataset_info）
        out["tags"] = [t for t in info.get("tags", [])
                       if not t.startswith(("region", "size_categories"))]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:90]}"
    return out


print("=" * 104)
print("困难任务候选数据集探测（用于泛化性实验）")
print("=" * 104)
print("筛选标准：困难（非饱和）+ 与反讽同族（语用/隐含意义）+ 二分类优先 + 规模 ≤5000 + 标注可靠")
print("=" * 104)

results = []
for ds_id, family, size, why in CANDIDATES:
    r = probe(ds_id)
    r.update({"family": family, "size_hint": size, "why_hard": why})
    results.append(r)
    status = "存在" if r["exists"] else "不可用"
    print(f"\n{'─'*104}")
    print(f"【{status}】{ds_id}")
    print(f"  任务族: {family}   规模提示: {size}")
    print(f"  难度依据: {why}")
    if not r["exists"]:
        print(f"  错误: {r.get('error')}")
        continue
    print(f"  下载量 {r.get('downloads')}  点赞 {r.get('likes')}  文件数 {r.get('n_files')}")
    if r.get("files"):
        print(f"  文件: {', '.join(r['files'][:8])}")

# ── 排序推荐 ──
print(f"\n\n{'=' * 104}")
print("【推荐排序】（存在 + 任务族匹配 + 规模合适）")
print("=" * 104)
PRIORITY = {"A 上下文讽刺": 1, "A 对话讽刺": 1, "A FIGLANG": 2,
            "A 推特讽刺": 3, "B 隐含仇恨": 1, "B 仇恨言论": 4,
            "A 上下文讽刺(多模态)": 5, "B 仇恨": 4,
            "B 仇恨(子集)": 2, "B 参照(情感)": 9, "D 参照": 9, "D": 9}
avail = [r for r in results if r["exists"]]
avail.sort(key=lambda r: (PRIORITY.get(r["family"], 6), -int(r.get("downloads") or 0)))
print(f"  {'优先级':<6} {'数据集 ID':<52} {'任务族':<22} {'下载量':>7}")
print("  " + "-" * 96)
for i, r in enumerate(avail, 1):
    pr = PRIORITY.get(r["family"], 6)
    flag = "★" if pr <= 2 else ""
    print(f"  {pr:<6} {r['id']:<52} {r['family']:<22} "
          f"{r.get('downloads', 0):>7} {flag}")

os.makedirs("experiments/results", exist_ok=True)
with open("experiments/results/hard_task_candidates.json", "w", encoding="utf-8") as f:
    json.dump({"candidates": results,
               "recommended_order": [r["id"] for r in avail]},
              f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 experiments/results/hard_task_candidates.json")
print(f"\n说明：本脚本只探测【是否存在与元数据】，未下载内容、未核实真实规模与标注质量。")
print(f"      下一步需对 Top-3 候选逐个下载并统计标签分布、类别歧义度、测试集规模，")
print(f"      并核验标注机制是否可靠（避免重蹈 Misra 按新闻源标注、模型靠文风作弊的覆辙）。")
