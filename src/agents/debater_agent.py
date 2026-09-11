import re
from .base_agent import BaseAgent
from typing import Dict, Any, List, Optional

BINARY_VOTES = ["pro", "con"]
MULTICLASS_VOTES = ["world", "sports", "business", "tech"]

# 标签到展示词的映射（与裁决者词表一致）
BINARY_LABEL_NAMES = {0: "con", 1: "pro"}
MULTICLASS_LABEL_NAMES = {0: "world", 1: "sports", 2: "business", 3: "tech"}
# 反讽任务：示例展示用自然语言标签，投票仍用 pro（ironic）/con（not ironic）
IRONY_LABEL_NAMES = {0: "not ironic", 1: "ironic"}
# hate 任务：label=1 → hateful → pro 票；label=0 → not hateful → con 票
HATE_LABEL_NAMES = {0: "not hateful", 1: "hateful"}


def parse_vote(text: str, task: str = "sentiment") -> Optional[str]:
    """从辩论者的论据中解析其显式投票。
    优先解析 'FINAL ANSWER: <vote>' 标记；失败时回退到最后一个独立单词投票。
    二分类（情感/反讽）使用 pro/con 词表，与裁决者保持一致。"""
    keywords = BINARY_VOTES if task in ("sentiment", "irony", "hate") else MULTICLASS_VOTES
    t = text.lower()

    # 1) 优先匹配 FINAL ANSWER 标记（取最后一次出现）
    matches = re.findall(r"final\s*answer\s*[:：]\s*(\w+)", t)
    for m in reversed(matches):
        if m in keywords:
            return m
        if task in ("sentiment", "irony", "hate"):
            # 兼容 positive/negative、ironic/not ironic、hateful/not hateful 的写法
            if m in ("positive", "ironic", "hateful", "hate"):
                return "pro"
            if m == "negative":
                return "con"
            if task in ("irony", "hate") and m == "not":
                # 模型输出 "FINAL ANSWER: not ironic" / "not hateful" 时正则只捕获到 "not"
                return "con"

    # 2) 回退：最后一个作为独立单词出现的投票词（避免 "con" 误匹配 "counter" 等）
    last_pos, vote = -1, None
    for kw in keywords:
        for m in re.finditer(rf"\b{kw}\b", t):
            if m.start() > last_pos:
                last_pos, vote = m.start(), kw
    return vote


def format_examples(examples: List[Dict[str, Any]], task: str = "sentiment") -> str:
    """把检索到的少样本示例格式化为提示词片段。"""
    if not examples:
        return ""
    if task == "irony":
        label_names = IRONY_LABEL_NAMES
    elif task == "hate":
        label_names = HATE_LABEL_NAMES
    elif task == "sentiment":
        label_names = BINARY_LABEL_NAMES
    else:
        label_names = MULTICLASS_LABEL_NAMES
    lines = []
    for j, ex in enumerate(examples, 1):
        label_word = label_names.get(ex["label"], str(ex["label"]))
        lines.append(f"Example {j} (label: {label_word}): {ex['text']}")
    return "\n".join(lines)


class DebaterAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any], stance: str):
        super().__init__(config)
        self.stance = stance
        self.task = config.get("task", "sentiment")

    def generate_argument(self, sample: Dict[str, Any], context: str = "",
                          examples: Optional[List[Dict[str, Any]]] = None) -> str:
        text = sample['text']
        examples_block = format_examples(examples, self.task) if examples else ""
        examples_section = (
            f"Here are some similar labeled examples for reference:\n{examples_block}\n\n"
            if examples_block else ""
        )
        if self.task == "sentiment":
            stance_desc = "POSITIVE" if self.stance == "pro" else "NEGATIVE"
            prompt = (
                f"{examples_section}"
                f"The following movie review is believed to express a {stance_desc} sentiment. "
                f"Argue in support of this view, and address possible counterarguments.\n"
                f"Review: {text}\n"
                f"You may reconsider after seeing the opponent's arguments. "
                f"End your argument with exactly one line: 'FINAL ANSWER: pro' or 'FINAL ANSWER: con'."
            )
        elif self.task == "irony":
            stance_desc = "IRONIC" if self.stance == "pro" else "NOT IRONIC"
            prompt = (
                f"{examples_section}"
                f"The following tweet is believed to be {stance_desc}. "
                f"Argue in support of this view using clues like exaggeration, sarcasm, emoji, or contradiction between literal and intended meaning. Address possible counterarguments.\n"
                f"Tweet: {text}\n"
                f"You may reconsider after seeing the opponent's arguments. "
                f"End your argument with exactly one line: 'FINAL ANSWER: pro' (ironic) or 'FINAL ANSWER: con' (not ironic)."
            )
        else:
            categories = "world (world news), sports (sports news), business (business news), or technology (tech news)"
            if self.stance == "pro":
                prompt = (
                    f"{examples_section}"
                    f"Classify the following news into one of these categories: {categories}.\n"
                    f"News: {text}\n"
                    f"Provide a rationale, then end with exactly one line: 'FINAL ANSWER: <category>'."
                )
            else:
                prompt = (
                    f"{examples_section}"
                    f"Argue AGAINST the most obvious classification for the following news, and suggest a different category from: {categories}.\n"
                    f"News: {text}\n"
                    f"Provide a rationale, then end with exactly one line: 'FINAL ANSWER: <category>'."
                )
        if context:
            prompt = f"{context}\n\n{prompt}"
        return self.generate(prompt, max_new_tokens=80)

    def _lens(self) -> str:
        """视角分化提示词（核心修复）。

        设计原则：给双方不同的「审查视角」而非不同的「结论」。视角决定关注哪类证据，
        结论仍由模型自主给出。这样既产生真实的意见多样性，又不会退化成恒定输出的
        投票器——旧实现两方提示词逐字相同，导致 98.7% 同票、判决退化为常数预测器。
        """
        if self.task == "irony":
            # ⚠️ 两方认知负担必须对称。旧版给正方加了"必须列出具体标记否则判字面"的
            # 强举证要求，1.5B 模型执行不了这种元指令，结果正方投 pro 仅 28%、
            # 反方反而 46%——视角与行为完全倒挂。现改为对称的"从X角度审视后判断"。
            if self.stance == "pro":
                return (
                    "Your assigned review lens: IRONY DETECTION.\n"
                    "Read the tweet and consider whether the wording is meant ironically — "
                    "for example exaggeration, sarcasm, positive words describing a clearly "
                    "negative situation, or an emoji/hashtag that clashes with the text.\n"
                    "Say briefly what you notice, then give your verdict."
                )
            return (
                "Your assigned review lens: LITERAL READING.\n"
                "Read the tweet and consider whether it simply means what it says — "
                    "a sincere statement, a plain description, or a straightforward opinion "
                    "with no hidden opposite meaning.\n"
                "Say briefly what you notice, then give your verdict."
            )
        if self.task == "hate":
            # ⚠️ 严格对称设计，直接吸取 irony 视角倒挂的教训：
            #    irony 任务中正方拿到 7 类可检验证据、反方只有 4 个抽象形容词
            #    （sincere/plain/straightforward），导致反方（本应更保守）投 pro 率
            #    反而高出 15.3pp —— 视角与行为完全倒挂（见 diagnose_lens_inversion.py）。
            #    故此处两方都给【具体可检验的线索类型】，条数与抽象层级对齐；
            #    且反方不使用 "no hidden/opposite meaning" 这类否定启动表述
            #    （提及即激活，反而提示模型去找攻击性）。
            #    显性 vs 隐性攻击是 hate 任务的真实难点（隐性攻击表层无脏词），
            #    两个视角都有独立发现能力，不构成"一方有约束一方无约束"。
            if self.stance == "pro":
                return (
                    "Your assigned review lens: EXPLICIT HOSTILITY DETECTION.\n"
                    "Read the tweet and consider whether it directly attacks or demeans a "
                    "group or person — for example slurs or dehumanizing words, threats of "
                    "violence, calls to exclude or harm, or insults aimed at identity "
                    "(race, gender, religion, origin, orientation).\n"
                    "Say briefly what you notice, then give your verdict."
                )
            return (
                "Your assigned review lens: IMPLICIT HOSTILITY DETECTION.\n"
                "Read the tweet and consider whether it attacks or demeans indirectly — for "
                "example a mocking or contemptuous tone, stereotyping, false praise that "
                "belittles, rhetorical questions implying inferiority, or quoting someone "
                "in order to ridicule them.\n"
                "Say briefly what you notice, then give your verdict."
            )
        if self.task == "sentiment":
            if self.stance == "pro":
                return (
                    "Your assigned review lens: PRAISE DETECTION.\n"
                    "Identify every element the reviewer praises or enjoys, then judge whether that "
                    "praise is genuine or undercut elsewhere in the review."
                )
            return (
                "Your assigned review lens: CRITICISM DETECTION.\n"
                    "Identify every flaw, weakness or disappointment the reviewer mentions, then judge "
                    "whether that criticism dominates the review or is outweighed by positive remarks."
            )
        if self.stance == "pro":
            return (
                "Your assigned lens: KEYWORD ANALYSIS.\n"
                "Identify the dominant entities and topic keywords, and decide which category they "
                "most directly indicate."
            )
        return (
            "Your assigned lens: ALTERNATIVE READING.\n"
            "Name the category that looks most obvious at first glance, then examine whether a less "
            "obvious category actually fits the content better."
        )

    def _classify_prompt(self, sample: Dict[str, Any],
                         examples: Optional[List[Dict[str, Any]]] = None) -> str:
        """构造带视角分化的分类提示词（逐条与批量共用，保证结果一致）。"""
        text = sample['text']
        examples_block = format_examples(examples, self.task) if examples else ""
        examples_section = (
            f"Here are some labeled reference examples:\n{examples_block}\n\n"
            if examples_block else ""
        )
        lens = self._lens()
        if self.task == "sentiment":
            return (
                f"{examples_section}{lens}\n\n"
                f"Review: {text}\n"
                f"Apply your lens, then decide the overall sentiment as 'pro' (positive) or 'con' (negative). "
                f"End with exactly one line: 'FINAL ANSWER: pro' or 'FINAL ANSWER: con'."
            )
        if self.task == "irony":
            return (
                f"{examples_section}{lens}\n\n"
                f"Tweet: {text}\n"
                f"Apply your lens, then decide whether the tweet is ironic: 'pro' if IRONIC "
                f"(says the opposite of what it means), 'con' if NOT IRONIC (literal, sincere). "
                f"End with exactly one line: 'FINAL ANSWER: pro' or 'FINAL ANSWER: con'."
            )
        if self.task == "hate":
            return (
                f"{examples_section}{lens}\n\n"
                f"Tweet: {text}\n"
                f"Apply your lens, then decide whether the tweet is hateful: 'pro' if HATEFUL "
                f"(attacks or demeans a group or person, explicitly or implicitly), "
                f"'con' if NOT HATEFUL (critical but not attacking a group or person). "
                f"End with exactly one line: 'FINAL ANSWER: pro' or 'FINAL ANSWER: con'."
            )
        categories = "world (world news), sports (sports news), business (business news), or technology (tech news)"
        return (
            f"{examples_section}{lens}\n\n"
            f"News: {text}\n"
            f"Apply your lens, then choose one category from: {categories}. "
            f"End with exactly one line: 'FINAL ANSWER: <category>'."
        )

    def analyze(self, sample: Dict[str, Any],
                examples: Optional[List[Dict[str, Any]]] = None):
        """带视角分化的独立判断，返回 (投票, 原始论据文本)。
        返回文本是为了后续交叉质询时能把对方的真实论证作为上下文传入。"""
        prompt = self._classify_prompt(sample, examples)
        response = self.generate(prompt, max_new_tokens=100)
        return parse_vote(response, self.task), response

    def analyze_batch(self, samples: List[Dict[str, Any]],
                      examples_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
                      batch_size: int = 8):
        """批量视角分化判断，返回 (投票列表, 原始文本列表)。服务器大显存下用。"""
        if examples_list is None:
            examples_list = [None] * len(samples)
        prompts = [self._classify_prompt(s, ex) for s, ex in zip(samples, examples_list)]
        votes, texts = [], []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            responses = self.generate_batch(chunk, max_new_tokens=100)
            texts.extend(responses)
            votes.extend(parse_vote(r, self.task) for r in responses)
        return votes, texts

    def classify(self, sample: Dict[str, Any],
                 examples: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """中立的上下文学习（ICL）分类：不预设立场，依据检索到的示例对样本分类。
        examples 为空时即零样本分类。返回解析后的投票标签（pro/con 或类别名）。"""
        prompt = self._classify_prompt(sample, examples)
        response = self.generate(prompt, max_new_tokens=100)
        return parse_vote(response, self.task)

    def classify_batch(self, samples: List[Dict[str, Any]],
                       examples_list: Optional[List[Optional[List[Dict[str, Any]]]]] = None,
                       batch_size: int = 8) -> List[Optional[str]]:
        """批量中立分类：服务器大显存下用，逐条版 classify_batch(batch_size=1) 等价。"""
        if examples_list is None:
            examples_list = [None] * len(samples)
        prompts = [self._classify_prompt(s, ex) for s, ex in zip(samples, examples_list)]
        votes = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            responses = self.generate_batch(chunk, max_new_tokens=100)
            votes.extend(parse_vote(r, self.task) for r in responses)
        return votes

    def defend_judgment(self, sample: Dict[str, Any], own_vote: Optional[str],
                        opponent_vote: Optional[str], context: str = "",
                        examples: Optional[List[Dict[str, Any]]] = None) -> str:
        """交叉质询：检验并（必要时）修正自己先前的独立判断。

        own_vote / opponent_vote 为双方独立判断阶段的结论；context 为对方论据文本。
        提示词如实描述双方是一致还是分歧——不能写死"对方不同意"，否则一致时会向模型
        传入虚假前提，诱导其强行制造分歧。
        """
        text = sample['text']
        examples_block = format_examples(examples, self.task) if examples else ""
        examples_section = (
            f"Here are some similar labeled examples for reference:\n{examples_block}\n\n"
            if examples_block else ""
        )
        lens = self._lens()
        if self.task == "irony":
            names = {"pro": "IRONIC", "con": "NOT IRONIC"}
            task_desc = "whether the tweet is ironic"
            final_line = ("End your response with exactly one line: "
                          "'FINAL ANSWER: pro' (ironic) or 'FINAL ANSWER: con' (not ironic).")
        elif self.task == "hate":
            names = {"pro": "HATEFUL", "con": "NOT HATEFUL"}
            task_desc = "whether the tweet is hateful"
            final_line = ("End your response with exactly one line: "
                          "'FINAL ANSWER: pro' (hateful) or 'FINAL ANSWER: con' (not hateful).")
        elif self.task == "sentiment":
            names = {"pro": "POSITIVE", "con": "NEGATIVE"}
            task_desc = "the sentiment of the movie review"
            final_line = ("End your response with exactly one line: "
                          "'FINAL ANSWER: pro' or 'FINAL ANSWER: con'.")
        else:
            names = {c: c for c in MULTICLASS_VOTES}
            task_desc = "the news category"
            final_line = ("End your response with exactly one line: "
                          "'FINAL ANSWER: <category>'.")
        own_desc = names.get(own_vote, own_vote or "undecided")
        opp_desc = names.get(opponent_vote, opponent_vote or "undecided")
        opp_section = f"\nTheir reasoning:\n{context}\n" if context else ""

        # 如实描述双方关系：一致 / 分歧 / 对方未表态
        if opponent_vote is None:
            relation = ("Another reviewer did not produce a clear verdict."
                        " Re-examine your own judgment on its merits.")
        elif own_vote == opponent_vote:
            relation = (
                f"Another reviewer, using a different lens, reached the SAME conclusion "
                f"({opp_desc}). Agreement is not proof of correctness — independently check "
                f"whether your evidence actually supports it, and state what would falsify it."
            )
        else:
            relation = (
                f"Another reviewer, using a different lens, DISAGREED and answered "
                f"{opp_desc}. Examine their reasoning and decide which evidence is stronger."
            )

        prompt = (
            f"{examples_section}{lens}\n\n"
            f"Earlier you judged {task_desc} for the text below as: {own_desc}.\n"
            f"{relation}{opp_section}\n"
            f"Text: {text}\n"
            f"Re-examine with concrete evidence from the text. Keep your judgment only if the "
            f"evidence holds; change it if the evidence does not. "
            f"{final_line}"
        )
        return self.generate(prompt, max_new_tokens=100)
