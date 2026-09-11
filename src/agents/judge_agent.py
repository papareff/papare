import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Optional


class JudgeAgent:
    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("name", "distilbert-base-uncased-finetuned-sst-2-english")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("max_length", 128)
        # 投票权重：三方等权（多数表决）。若裁决者权重≥两位辩论者之和，
        # 辩论将永远无法改变结果，消融实验会失去意义。
        self.judge_weight = config.get("judge_weight", 1.0)
        self.debater_weight = config.get("debater_weight", 1.0)
        # 模块 IV：置信度门控。gate_threshold=None 表示关闭门控（始终辩论）。
        # 依据：实测辩论仅在裁决者低置信时有正增益（反讽 conf<0.6 子集 +6.78pp），
        # 在高置信时严重有害（0.7~0.9 子集 −29.55pp）。
        self.gate_threshold = config.get("gate_threshold")
        # 投票门槛：判为「pro/反讽」所需的最少票数（三方各 1 票，满分 3）。
        #   2 = 多数票（默认，保持与历史实验完全一致）
        #   3 = 全票制（实测偏置修复最强：n=784 中期 65.49%→75.00%，
        #       判 pro 率 64.9%→22.8%，偏置 +26.4pp→−15.8pp）
        # 依据 scripts/diagnose_lens_inversion.py 的跨数据集验证：
        #   ≥3 票在 n=200 自然集与 n=784 中期集上均优于纯裁决者，平均 +5.91pp。
        # 背景：辩论方存在系统性 pro 偏置（LR_pro 1.73 / LR_con 1.35），
        #   提高门槛等价于要求更强证据才判反讽，从而抵消偏置。
        self.vote_threshold = int(config.get("vote_threshold", 2))
        if self.vote_threshold < 1:
            self.vote_threshold = 1
        self._classify_cache = {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        # 允许通过 config 显式指定标签映射（如反讽任务微调模型）
        cfg_label_map = config.get("id_to_label")
        if cfg_label_map is not None:
            # config 里 key 可能是 str，统一转 int
            self.id_to_label = {int(k): v for k, v in cfg_label_map.items()}
            self.mode = "binary" if self.model.config.num_labels == 2 else "multiclass"
        elif "sst-2" in self.model_name:
            # 二分类使用 pro/con 词表，与辩论投票和实验脚本保持一致
            self.id_to_label = {0: "con", 1: "pro"}
            self.mode = "binary"
        elif "ag-news" in self.model_name or self.model.config.num_labels == 4:
            # AG News: 0-World, 1-Sports, 2-Business, 3-Tech
            self.id_to_label = {0: "world", 1: "sports", 2: "business", 3: "tech"}
            self.mode = "multiclass"
        elif self.model.config.num_labels == 2:
            # 其他二分类模型（如反讽）默认 pro/con 词表：0=con, 1=pro
            self.id_to_label = {0: "con", 1: "pro"}
            self.mode = "binary"
        else:
            self.id_to_label = {i: str(i) for i in range(self.model.config.num_labels)}
            self.mode = "multiclass"
        print(f"JudgeAgent loaded {self.model_name} on {self.device} in {self.mode} mode")

    def _classify(self, text: str) -> Dict[str, Any]:
        """对干净文本做独立分类，返回预测类别与置信度。

        带缓存：裁决者是冻结的确定性模型（eval + no_grad），同一文本结果恒定。
        pipeline 每条样本会调用 _classify 2~3 次（门控判定、make_decision），
        缓存可消除重复前向。缓存键含 max_length 以防配置变更导致串味。
        """
        key = (text, self.max_length)
        cached = self._classify_cache.get(key)
        if cached is not None:
            return cached
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = torch.argmax(probs, dim=-1).item()
        result = {
            "pred_id": pred,
            "pred_label": self.id_to_label.get(pred, "unknown"),
            "confidence": probs[pred].item(),
            "prob_vector": probs.tolist(),
        }
        self._classify_cache[key] = result
        return result

    def classify_sample(self, text: str) -> Dict[str, Any]:
        """公开的分类入口，供路由器等外部组件获取裁决者置信度。

        与 _classify 同源（带缓存），因此路由阶段取到的置信度与后续
        门控判定、make_decision 完全一致，不会产生同一文本多次前向的开销，
        也不会出现「路由用的置信度」与「计票用的置信度」不一致的问题。
        """
        return self._classify(text)

    def should_debate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """门控判定（模块 IV）：仅当裁决者置信度低于阈值时才需要启动辩论。
        返回 {"debate": bool, "confidence": float, "judge_vote": str}。
        关闭门控（gate_threshold=None）时恒返回 True。"""
        r = self._classify(sample['text'])
        if self.gate_threshold is None:
            return {"debate": True, "confidence": r["confidence"], "judge_vote": r["pred_label"]}
        return {
            "debate": r["confidence"] < self.gate_threshold,
            "confidence": r["confidence"],
            "judge_vote": r["pred_label"],
        }

    def make_decision(self, sample: Dict[str, Any], debate_votes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        辩论投票制裁决（含模块 IV 置信度门控 + 票数门槛）：
        1. 裁决者仅基于原始文本独立分类（不再拼接辩论历史，避免情感词污染）；
        2. 门控开启且置信度 >= 阈值时，直接采纳裁决者判决，忽略辩论投票；
        3. 否则收集正反双方辩论者的显式投票并加权计票；
        4. 二分类下按 vote_threshold 决定判 pro 所需票数（默认 2=多数票，3=全票制）；
        5. 多分类仍用「得票最高 + 平票时裁决者打破」。

        ⚠️ 基线保护：debate_votes 为空（纯裁决者基线调用）时【不套用票数门槛】，
           直接返回裁决者判决。否则 ≥3 票门槛会把基线全部压成 con，破坏对照语义。
        返回包含 decision（str）、gated（bool）、投票细节与计票中间量的字典。
        """
        judge_result = self._classify(sample['text'])
        judge_vote = judge_result["pred_label"]
        confidence = judge_result["confidence"]

        # 模块 IV：高置信样本短路，不采纳辩论意见
        gated = self.gate_threshold is not None and confidence >= self.gate_threshold
        if gated:
            return {
                "decision": judge_vote,
                "judge_vote": judge_vote,
                "judge_confidence": confidence,
                "gated": True,
                "debate_votes": {},
                "vote_detail": [{"voter": "judge", "vote": judge_vote, "weight": self.judge_weight}],
                "score": {judge_vote: self.judge_weight},
                "vote_threshold": self.vote_threshold,
                "pro_score": None,
                "required_pro_score": None,
            }

        # ── 基线保护：无辩论票时直接采纳裁决者判决 ──
        valid_votes = {k: v for k, v in (debate_votes or {}).items() if v is not None}
        if not valid_votes:
            return {
                "decision": judge_vote,
                "judge_vote": judge_vote,
                "judge_confidence": confidence,
                "gated": False,
                "debate_votes": debate_votes or {},
                "vote_detail": [{"voter": "judge", "vote": judge_vote, "weight": self.judge_weight}],
                "score": {judge_vote: self.judge_weight},
                "vote_threshold": self.vote_threshold,
                "pro_score": None,
                "required_pro_score": None,
                "baseline_no_debate": True,
            }

        # ── 计票 ──
        scores: Dict[str, float] = {}
        votes_cast: List[Dict[str, Any]] = [
            {"voter": "judge", "vote": judge_vote, "weight": self.judge_weight}]
        scores[judge_vote] = scores.get(judge_vote, 0.0) + self.judge_weight
        for voter, vote in valid_votes.items():
            votes_cast.append({"voter": voter, "vote": vote, "weight": self.debater_weight})
            scores[vote] = scores.get(vote, 0.0) + self.debater_weight

        # ── 二分类：票数门槛（偏置修复的核心开关）──
        # 门槛按比例缩放，兼容不等权与缺票：
        #   ratio = vote_threshold / 3（等权三方满票为 3）
        #   required = ratio × 当前可能的最大总权重
        # 等权齐全时：threshold=2 → required=2（多数票）；threshold=3 → required=3（全票）
        # 缺票时：n_valid=2 → total=2，threshold=3 → required=2（两票都需一致）
        if self.mode == "binary" and "pro" in self.id_to_label.values():
            pro_score = scores.get("pro", 0.0)
            total_possible = (self.judge_weight
                              + len(valid_votes) * self.debater_weight)
            ratio = self.vote_threshold / 3.0
            required = ratio * total_possible
            if pro_score >= required:
                decision = "pro"
            else:
                # 判 con。平票/边界情况由裁决者意见作为参考记录在 vote_detail
                decision = "con"
            return {
                "decision": decision,
                "judge_vote": judge_vote,
                "judge_confidence": confidence,
                "gated": False,
                "debate_votes": debate_votes or {},
                "vote_detail": votes_cast,
                "score": scores,
                "vote_threshold": self.vote_threshold,
                "pro_score": round(pro_score, 4),
                "required_pro_score": round(required, 4),
                "n_valid_votes": len(valid_votes) + 1,
            }

        # ── 多分类：得票最高，平票时裁决者打破 ──
        decision = judge_vote
        if scores:
            best_score = max(scores.values())
            winners = [k for k, v in scores.items() if v == best_score]
            if len(winners) == 1:
                decision = winners[0]
            elif judge_vote in winners:
                decision = judge_vote
            else:
                decision = winners[0]

        return {
            "decision": decision,
            "judge_vote": judge_vote,
            "judge_confidence": confidence,
            "gated": False,
            "debate_votes": debate_votes or {},
            "vote_detail": votes_cast,
            "score": scores,
            "vote_threshold": self.vote_threshold,
            "pro_score": None,
            "required_pro_score": None,
        }
