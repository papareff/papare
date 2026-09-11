import random
from typing import Dict, Any, Optional


class SampleAwareRouter:
    """样本感知路由器：根据样本特征决定推理策略（零样本 / 少样本）。

    模式：
      • length_based（历史默认，保留兼容）
          len(text.split()) > length_threshold → few_shot
          ⚠️ 对短文本任务失效：反讽推文词数均值 14.6 / 最大 31，
             而 length_threshold=100 → 触发率恒为 0.00%，few_shot 分支是死代码。
             诊断依据：scripts/diagnose_lens_inversion.py【Q3】

      • confidence_based（推荐，用于反讽这类短文本难任务）
          裁决者置信度 < confidence_threshold → few_shot（难样本，检索示例辅助）
          否则 → zero_shot（易样本，无需示例）
          依据：conf<0.90 子集裁决者准确率仅 59.43%，conf≥0.90 达 73.42%，
                说明置信度能有效区分难易，难样本正是检索增强该介入的地方。
          ⚠️ 需要上游传入裁决者置信度；未传入时回退到 length_based 判定。

      • disabled：恒返回 zero_shot，不做任何路由决策。
    """

    def __init__(self, config: Dict[str, Any]):
        self.mode = config.get("mode", "length_based")
        self.length_threshold = config.get("length_threshold", 100)
        self.confidence_threshold = config.get("confidence_threshold", 0.90)
        # 路由判定为 few_shot 时检索的示例数量。
        # 仅当调用方未显式指定 num_shots（或为 0）时生效。
        self.few_shot_k = config.get("few_shot_k", 3)
        self.few_shot_examples = []  # 可以加载一些示例
        self.last_decision: Dict[str, Any] = {}

    def route(self, sample: Dict[str, Any],
              confidence: Optional[float] = None) -> str:
        """根据样本特征（及可选的裁决者置信度）返回路由策略。

        返回 "zero_shot" / "few_shot"。
        """
        if self.mode == "disabled":
            self._record("disabled", confidence, "zero_shot")
            return "zero_shot"

        if self.mode == "confidence_based":
            if confidence is not None:
                strat = "few_shot" if confidence < self.confidence_threshold else "zero_shot"
                self._record("confidence_based", confidence, strat)
                return strat
            # 未提供置信度时回退到长度判定，避免静默返回全 zero_shot
            strat = self._length_rule(sample)
            self._record("confidence_based(fallback:length)", confidence, strat)
            return strat

        if self.mode == "length_based":
            strat = self._length_rule(sample)
            self._record("length_based", confidence, strat)
            return strat

        if self.mode == "category_based":
            label = sample.get("label", 0)
            strat = "zero_shot" if label in [0, 1] else "few_shot"
            self._record("category_based", confidence, strat)
            return strat

        if self.mode == "random":
            strat = random.choice(["zero_shot", "few_shot"])
            self._record("random", confidence, strat)
            return strat

        self._record("unknown_mode→zero_shot", confidence, "zero_shot")
        return "zero_shot"

    def _length_rule(self, sample: Dict[str, Any]) -> str:
        text = sample.get("text", "")
        return "few_shot" if len(text.split()) > self.length_threshold else "zero_shot"

    def _record(self, mode_used: str, confidence: Optional[float], strategy: str):
        """记录本次路由决策，供实验脚本落盘审计。"""
        self.last_decision = {
            "mode_used": mode_used,
            "confidence": confidence,
            "confidence_threshold": self.confidence_threshold,
            "length_threshold": self.length_threshold,
            "strategy": strategy,
        }

    def get_few_shot_examples(self, sample: Dict[str, Any]) -> list:
        """返回预置的少样本示例（第一阶段实现；检索式示例由 FewShotRetriever 提供）。"""
        return self.few_shot_examples[:3]

    def set_few_shot_examples(self, examples: list):
        self.few_shot_examples = examples
