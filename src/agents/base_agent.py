import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("name", "gpt2")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("max_length", 512)
        self.temperature = config.get("temperature", 0.7)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        # 支持共享已加载的模型/分词器（如正反方辩论者共用同一份模型，节省显存）
        shared_model = self.config.get("_shared_model")
        shared_tokenizer = self.config.get("_shared_tokenizer")
        if shared_model is not None and shared_tokenizer is not None:
            logger.info("Reusing shared model for agent")
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            return
        logger.info(f"Loading model {self.model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 大模型（如 7B）在小显存机器上需要限制显存占用、把溢出的层放到内存
        load_kwargs = dict(
            torch_dtype=torch.float16,   # 节省显存
            device_map="auto",           # 自动分配到GPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        max_memory = self.config.get("max_memory")
        if max_memory:
            load_kwargs["max_memory"] = max_memory
            logger.info(f"Using max_memory={max_memory}")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self.model.eval()
        self._input_device = self.model.get_input_embeddings().weight.device

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 跨设备放置（部分层在内存）时 model.device 不可靠，统一用词嵌入层所在设备
        input_device = getattr(self, "_input_device", None) or self.model.device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(input_device)
        temp = kwargs.get("temperature", self.temperature)
        # temperature=0 表示贪心解码，此时不能启用采样（transformers 会报错）
        do_sample = kwargs.get("do_sample", temp > 0)
        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", 50),  # 辩论论据适度
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temp
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_batch(self, prompts, **kwargs):
        """批量生成：一次前向处理多条提示，服务器大显存下可大幅提速。

        ⚠️ 解码器模型批量推理必须左填充（left padding），否则生成内容会错位。
        本方法临时切换 padding_side 并在结束后恢复，不影响其他调用。
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], **kwargs)]

        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        prev_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            model_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            )
        finally:
            self.tokenizer.padding_side = prev_side

        input_device = getattr(self, "_input_device", None) or self.model.device
        model_inputs = {k: v.to(input_device) for k, v in model_inputs.items()}

        temp = kwargs.get("temperature", self.temperature)
        do_sample = kwargs.get("do_sample", temp > 0)
        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", 50),
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temp

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        # 左填充下所有输入等长，统一用输入长度截掉前缀
        in_len = model_inputs["input_ids"].shape[1]
        gen_only = generated_ids[:, in_len:]
        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
