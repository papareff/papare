from src.router.sample_aware_router import SampleAwareRouter
from src.agents.debater_agent import DebaterAgent, parse_vote
from src.agents.judge_agent import JudgeAgent
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, config: Dict[str, Any], use_debate: bool = True,
                 retriever: Optional[Any] = None, num_shots: int = 0):
        self.config = config
        self.router = SampleAwareRouter(config.get("router", {}))
        self.use_debate = use_debate
        self.task = config.get("task", "sentiment")
        # 少样本检索器（模块 II）与 shot 数；num_shots=0 表示零样本
        self.retriever = retriever
        self.num_shots = num_shots

        # 仅在启用辩论时初始化辩论智能体（正反方共享同一份模型，节省显存）
        if self.use_debate:
            debate_cfg = self.config.get("debate", {}) or {}
            debater_config = {
                "name": debate_cfg.get("debater_model", "Qwen/Qwen2-1.5B-Instruct"),
                "device": "cuda",
                "max_length": 512,
                # 默认贪心解码（temperature=0）以保证结果可复现。
                # 双方的意见多样性来自不同的视角提示词，不需要靠采样噪声制造差异；
                # 用采样会使同一样本每次运行结果不同，n=50 上无法区分提示词改进与随机波动。
                "temperature": debate_cfg.get("temperature", 0.0),
                "task": self.task,
            }
            # 大模型（如 7B）在小显存机器上：限制显存，溢出层放内存
            if debate_cfg.get("max_memory"):
                debater_config["max_memory"] = debate_cfg["max_memory"]
            self.pro_debater = DebaterAgent(debater_config, stance="pro")
            # 反方复用正方已加载的模型与分词器，避免重复占用显存
            con_config = dict(debater_config)
            con_config["_shared_model"] = self.pro_debater.model
            con_config["_shared_tokenizer"] = self.pro_debater.tokenizer
            self.con_debater = DebaterAgent(con_config, stance="con")
        else:
            self.pro_debater = None
            self.con_debater = None

        # 裁决者始终加载（分类器），模型名可通过 config['judge'] 覆盖（如反讽微调模型）
        judge_config = {
            "name": "distilbert-base-uncased-finetuned-sst-2-english",
            "device": "cuda",
            "max_length": 512  # 增加以适应长文本
        }
        judge_config.update(self.config.get("judge", {}) or {})
        # 模块 IV 门控阈值：优先取 judge 配置，其次取 debate 配置
        gate = self.config.get("debate", {}).get("gate_threshold")
        if "gate_threshold" not in judge_config and gate is not None:
            judge_config["gate_threshold"] = gate
        # 投票门槛（偏置修复）：同样支持从 debate 配置传入。
        # 默认 2=多数票，与全部历史实验一致；3=全票制，实测跨集平均 +5.91pp。
        vt = self.config.get("debate", {}).get("vote_threshold")
        if "vote_threshold" not in judge_config and vt is not None:
            judge_config["vote_threshold"] = vt
        self.judge = JudgeAgent(judge_config)

    def _resolve_examples(self, sample: Dict[str, Any],
                          confidence: Optional[float]) -> tuple:
        """样本级路由（模块 II 接线）：决定 zero_shot / few_shot 并取回示例。

        优先级（保证与历史行为完全兼容）：
          1. retriever 为 None → 无从检索，沿用 num_shots 标注策略（历史行为）
          2. num_shots > 0    → 显式指定 k-shot，直接按 num_shots 检索（历史行为）
          3. num_shots == 0   → 交给路由器逐样本决策：
               confidence_based：裁决者置信度 < 阈值 → few_shot（难样本才检索）
               length_based    ：文本词数 > 阈值 → few_shot
                                 ⚠️ 反讽推文词数最大 31，阈值 100 → 触发率恒为 0
               disabled        ：恒 zero_shot
          路由器选 few_shot 时用 router.few_shot_k 作为检索条数。
        """
        if self.retriever is None:
            strategy = f"{self.num_shots}_shot" if self.num_shots > 0 else "zero_shot"
            return strategy, None, {"source": "no_retriever", "strategy": strategy}

        if self.num_shots > 0:
            examples = self.retriever.retrieve(sample['text'], k=self.num_shots)
            strategy = f"{self.num_shots}_shot"
            return strategy, examples, {"source": "explicit_num_shots",
                                        "strategy": strategy, "k": self.num_shots}

        strat = self.router.route(sample, confidence=confidence)
        info = dict(self.router.last_decision)
        info["source"] = "router"
        if strat == "few_shot":
            k = self.router.few_shot_k
            examples = self.retriever.retrieve(sample['text'], k=k)
            info["k"] = k
            info["strategy"] = f"{k}_shot"
            return f"{k}_shot", examples, info
        info["strategy"] = "zero_shot"
        return "zero_shot", None, info

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample['text']

        # 一次前向取裁决者判决与置信度。_classify 带缓存，
        # 故门控判定、样本级路由、最终计票三处共用这一次前向，无额外开销，
        # 也不会出现「路由所用置信度」与「计票所用置信度」不一致的问题。
        judge_pre = self.judge.classify_sample(text)
        confidence = judge_pre["confidence"]
        judge_vote_pre = judge_pre["pred_label"]

        # 模块 IV：置信度门控【前置】判定。
        # 高置信样本直接采纳裁决者判决，跳过检索与全部辩论生成（真实的算力节省）。
        # 门控置于路由之前，可避免为最终被跳过的样本白做一次 TF-IDF 检索。
        if self.judge.gate_threshold is not None and confidence >= self.judge.gate_threshold:
            return {
                "strategy": "zero_shot",
                "debate_history": [],
                "decision": judge_vote_pre,
                "judge_confidence": confidence,
                "judge_vote": judge_vote_pre,
                "gated": True,
                "debate_votes": {},
                "vote_detail": [{"voter": "judge", "vote": judge_vote_pre,
                                 "weight": self.judge.judge_weight}],
                "num_examples": 0,
                "route": {"source": "gated_skip", "strategy": "zero_shot",
                          "confidence": confidence,
                          "gate_threshold": self.judge.gate_threshold},
            }

        # 样本级路由（模块 II）：决定 zero_shot / few_shot 并检索示例
        strategy, examples, route_info = self._resolve_examples(sample, confidence)

        if not self.use_debate:
            # 无辩论：裁决者独立判决。
            # 注意门控已在上游前置执行，走到这里的样本必然未被门控跳过，
            # 且纯裁决者基线调用 make_decision(debate_votes=None) 受「基线保护」，
            # 不会套用票数门槛（否则 ≥3 票会把基线全部压成 con）。
            result = self.judge.make_decision(sample, debate_votes=None)
            return {
                "strategy": strategy,
                "debate_history": [],
                "decision": result["decision"],
                "judge_confidence": result["judge_confidence"],
                "judge_vote": result["judge_vote"],
                "gated": False,
                "vote_detail": result["vote_detail"],
                "num_examples": len(examples) if examples else 0,
                "route": route_info,
            }

        # 门控已前置执行：能走到这里说明 gate_threshold 为 None（关闭门控）
        # 或置信度低于阈值（需要辩论）。此处不再重复判定。

        # 多轮辩论
        debate_history = []
        debate_votes: Dict[str, str] = {"pro": None, "con": None}
        # 质询前的独立判断票。用于归因偏置来源：
        #   初始票就倒挂 → 视角提示词失效；质询后才倒挂 → 质询轮漂移。
        initial_votes: Dict[str, str] = {"pro": None, "con": None}
        rounds = self.config.get("debate", {}).get("rounds", 2)
        neutral = self.config.get("debate", {}).get("neutral", False)

        if neutral:
            # 视角分化辩论（修复版）：
            # 1) 双方用不同的审查视角独立判断（旧版两方提示词逐字相同，导致 98.7% 同票、
            #    判决退化为常数预测器）
            # 2) 无论是否一致都走交叉质询——一致时也要让对方检验己方证据，否则无法区分
            #    "真确信"与"同一模型重复采样"
            pro_initial, pro_arg0 = self.pro_debater.analyze(sample, examples=examples)
            con_initial, con_arg0 = self.con_debater.analyze(sample, examples=examples)
            debate_history.append(f"Pro (lens A) independent: {pro_initial}")
            debate_history.append(f"Con (lens B) independent: {con_initial}")
            debate_votes["pro"] = pro_initial
            debate_votes["con"] = con_initial
            initial_votes["pro"] = pro_initial
            initial_votes["con"] = con_initial

            pro_vote, con_vote = pro_initial, con_initial
            # 质询时把对方的真实论证文本作为上下文传入（而非旧版的空字符串）
            pro_ctx, con_ctx = con_arg0, pro_arg0
            for turn in range(rounds):
                pro_arg = self.pro_debater.defend_judgment(
                    sample, pro_vote, con_vote,
                    context=(pro_ctx or "").split("FINAL ANSWER")[0][:400],
                    examples=examples)
                con_arg = self.con_debater.defend_judgment(
                    sample, con_vote, pro_vote,
                    context=(con_ctx or "").split("FINAL ANSWER")[0][:400],
                    examples=examples)
                debate_history.append(f"Pro (round {turn + 1}): {pro_arg}")
                debate_history.append(f"Con (round {turn + 1}): {con_arg}")
                new_pro = parse_vote(pro_arg, self.task)
                new_con = parse_vote(con_arg, self.task)
                if new_pro:
                    pro_vote = new_pro
                if new_con:
                    con_vote = new_con
                # 下一轮互换上下文：各自阅读对方本轮的论证
                pro_ctx, con_ctx = con_arg, pro_arg
            debate_votes["pro"] = pro_vote
            debate_votes["con"] = con_vote
        else:
            # 预设立场辩论：每轮结束后解析双方立场
            context = ""
            for turn in range(rounds):
                pro_arg = self.pro_debater.generate_argument(sample, context, examples=examples)
                con_arg = self.con_debater.generate_argument(sample, context, examples=examples)
                debate_history.append(f"Pro (round {turn + 1}): {pro_arg}")
                debate_history.append(f"Con (round {turn + 1}): {con_arg}")
                context += f"\nPro: {pro_arg}\nCon: {con_arg}"
                # 每轮重新解析投票，取最后一轮的有效投票
                pro_vote = parse_vote(pro_arg, self.task)
                con_vote = parse_vote(con_arg, self.task)
                if pro_vote:
                    debate_votes["pro"] = pro_vote
                    # 首轮的有效票即「无质询上下文时」的初始立场
                    if turn == 0:
                        initial_votes["pro"] = pro_vote
                if con_vote:
                    debate_votes["con"] = con_vote
                    if turn == 0:
                        initial_votes["con"] = con_vote

        # 投票制裁决
        result = self.judge.make_decision(sample, debate_votes=debate_votes)
        return {
            "strategy": strategy,
            "debate_history": debate_history,
            "decision": result["decision"],
            "judge_confidence": result["judge_confidence"],
            "judge_vote": result["judge_vote"],
            "gated": result.get("gated", False),
            "debate_votes": debate_votes,
            # 独立判断阶段（质询前）的初始票。用于区分偏置来源：
            # 若初始票就已倒挂 → 视角提示词失效；若质询后才倒挂 → 质询轮漂移。
            # 此前所有 CSV 只记录质询后的票，导致无法归因，是已知数据缺口。
            "initial_votes": initial_votes,
            "vote_detail": result["vote_detail"],
            "num_examples": len(examples) if examples else 0,
            "route": route_info,
            # 计票审计：门槛与 pro 得分，便于事后复核票数门槛是否按预期生效
            "vote_threshold": result.get("vote_threshold"),
            "pro_score": result.get("pro_score"),
            "required_pro_score": result.get("required_pro_score"),
        }
