"""In-process vLLM shim for Haskins et al.'s gen_data.py (D-059).

Exposes the one method gen_data.py calls on an openai.AsyncOpenAI client:
    await client.chat.completions.create(model=, messages=, max_tokens=,
        temperature=, response_format=, extra_body=)
returning an object with .choices[0].message.content and
.choices[0].finish_reason.  No server, no port, no daemon: the engine lives in
this process and dies with it (molab-nodes skill, section 1).

The engine is built lazily inside the running event loop on the first call,
because gen_data.py constructs the client before asyncio.run().
"""
import asyncio
import re
import uuid
from types import SimpleNamespace

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


class VLLMChatClient:
    def __init__(self, model, *, max_model_len=32768, gpu_memory_utilization=0.85,
                 download_dir="/tmp/hf", enable_thinking=False, structured=False, seed=None):
        self.model = model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.download_dir = download_dir
        self.enable_thinking = enable_thinking
        self.structured = structured
        self.seed = seed
        self.engine = None
        self.tok = None
        self._lock = None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _ensure_engine(self):
        if self.engine is not None:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self.engine is not None:
                return
            from vllm import AsyncEngineArgs
            try:
                from vllm.v1.engine.async_llm import AsyncLLM as Engine
            except ImportError:  # older layout
                from vllm import AsyncLLMEngine as Engine
            from transformers import AutoTokenizer
            # vLLM 0.28: ModelConfig.seed must be an int (None fails pydantic
            # validation even though AsyncEngineArgs accepts it). vLLM's own
            # default is 0, which makes two runs of the same command produce
            # the same documents, so callers that need distinct samples MUST
            # pass distinct seeds (the corpus cell does).
            kw = dict(model=self.model, trust_remote_code=True,
                      max_model_len=self.max_model_len,
                      gpu_memory_utilization=self.gpu_memory_utilization,
                      download_dir=self.download_dir)
            if self.seed is not None:
                kw["seed"] = int(self.seed)
            args = AsyncEngineArgs(**kw)
            self.engine = Engine.from_engine_args(args)
            self.tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)

    def _sampling(self, max_tokens, temperature, response_format):
        from vllm import SamplingParams
        kw = dict(max_tokens=max_tokens, temperature=temperature)
        if self.structured and response_format:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                if response_format.get("type") == "json_schema":
                    kw["structured_outputs"] = StructuredOutputsParams(
                        json=response_format["json_schema"]["schema"])
                elif response_format.get("type") == "json_object":
                    kw["structured_outputs"] = StructuredOutputsParams(json_object=True)
            except Exception as e:  # fall back to free text; say so once, loudly
                if not getattr(self, "_warned_structured", False):
                    import sys
                    print("VLLM SHIM: structured outputs unavailable, falling back to free text:", repr(e)[:200], file=sys.stderr, flush=True)
                    self._warned_structured = True
        return SamplingParams(**kw)

    async def _create(self, *, messages, max_tokens, temperature, model=None,
                      response_format=None, extra_body=None, **_ignored):
        await self._ensure_engine()
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=self.enable_thinking)
        n_prompt = len(self.tok(prompt, add_special_tokens=False).input_ids)
        budget = max(64, min(int(max_tokens), self.max_model_len - n_prompt - 8))
        sp = self._sampling(budget, temperature, response_format)
        final = None
        async for out in self.engine.generate(prompt, sp, uuid.uuid4().hex):
            final = out
        o = final.outputs[0]
        text = _THINK_RE.sub("", o.text, count=1)
        self.calls += 1
        self.prompt_tokens += n_prompt
        self.completion_tokens += len(o.token_ids)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                     finish_reason=o.finish_reason)],
            usage=SimpleNamespace(prompt_tokens=n_prompt, completion_tokens=len(o.token_ids)),
        )
