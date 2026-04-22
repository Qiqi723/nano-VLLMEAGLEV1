# 在 nano-vLLM 上部署 EAGLE v1：左右源码对照版

> 这版文档不用 Markdown 表格嵌套代码块，因为很多 Markdown 渲染器会把表格里的 ```python 解析坏。下面改用 HTML 两栏布局：左栏是原始 nano-vLLM，右栏是接入 EAGLE 后的代码。每个需要对比的地方，页面都会被切成左右两半。

<style>
.compare-wrap {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 18px;
  margin: 18px 0 30px;
  align-items: stretch;
}
.compare-card {
  border: 1px solid #d8dee4;
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff;
}
.compare-title {
  padding: 10px 14px;
  font-weight: 700;
  border-bottom: 1px solid #d8dee4;
  background: #f6f8fa;
}
.compare-card pre {
  margin: 0;
  padding: 14px;
  overflow-x: auto;
  background: #f6f8fa;
  font-size: 13px;
  line-height: 1.45;
}
.compare-card code {
  font-family: Consolas, Monaco, 'Courier New', monospace;
  white-space: pre;
}
.note {
  border-left: 4px solid #57606a;
  padding: 8px 12px;
  background: #f6f8fa;
  color: #24292f;
}
@media (max-width: 900px) {
  .compare-wrap {
    grid-template-columns: 1fr;
  }
}
</style>

## 0. 目标

这次改造的目标是：在 nano-vLLM 上接入一个最小可用的 EAGLE v1 greedy speculative decoding 路径。

- 不传 `eagle_model`：保持原始 nano-vLLM 行为。
- 传入 `eagle_model` 且单序列、`temperature=0.0`：进入 EAGLE 路径。
- 当前版本先做 greedy、batch size = 1、tensor parallel = 1。
- 真实加速需要训练好的 EAGLE draft 权重；dummy draft 只能验证代码路径。

---

## 1. Config：增加 EAGLE 开关

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )
        assert self.max_num_batched_tokens >= self.max_model_len</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False

    # 保留本地量化开关，不影响 EAGLE
    enable_quant: bool = False
    quant_mode: str = "none"

    # EAGLE 开关
    eagle_model: str | None = None
    eagle_max_draft_tokens: int = 4

    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if self.eagle_model is not None:
            assert os.path.isdir(self.eagle_model)
            assert self.eagle_max_draft_tokens >= 1
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )
        assert self.max_num_batched_tokens >= self.max_model_len</code></pre>
  </div>
</div>

<div class="note">这里的关键点是：EAGLE 是显式开关。默认 `eagle_model=None`，普通 nano-vLLM 路径不会改变。</div>

---

## 2. SamplingParams / Sampler：允许 greedy decoding

EAGLE v1 最适合先从 greedy 做起，因为验证逻辑非常直接：candidate token 必须等于 target model 的 `argmax`。

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature &gt; 1e-10, \
            "greedy sampling is not permitted"


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs)
            .exponential_(1)
            .clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature &gt;= 0, \
            "temperature must be non-negative"


class Sampler(nn.Module):

    def forward(self, logits, temperatures):
        if torch.all(temperatures &lt;= 1e-10):
            return logits.argmax(dim=-1)

        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs)
            .exponential_(1)
            .clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens</code></pre>
  </div>
</div>

---

## 3. Qwen3Model：返回 EAGLE 需要的中间 feature

EAGLE draft model 不直接从 token ids 预测，而是利用 target model 的中间 feature。这里选择在进入最后一个 decoder layer 前保存 feature。

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">class Qwen3Model(nn.Module):

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">class Qwen3Model(nn.Module):

    def forward(
        self,
        input_ids,
        positions,
        return_eagle_features: bool = False,
    ):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        eagle_features = None

        for i, layer in enumerate(self.layers):
            if return_eagle_features and i == len(self.layers) - 1:
                eagle_features = (
                    hidden_states if residual is None
                    else hidden_states + residual
                )
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if return_eagle_features:
            return hidden_states, eagle_features
        return hidden_states


class Qwen3ForCausalLM(nn.Module):

    def forward(self, input_ids, positions, return_eagle_features=False):
        return self.model(input_ids, positions, return_eagle_features)</code></pre>
  </div>
</div>

---

## 4. 新增 `nanovllm/eagle.py`：EAGLE draft 模型

原始 nano-vLLM 没有 EAGLE draft 模型，所以这一块是新增文件。

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-text">nanovllm/
  config.py
  llm.py
  sampling_params.py
  engine/
    llm_engine.py
    model_runner.py
    scheduler.py
    block_manager.py
  models/
    qwen3.py

# 没有 eagle.py
# 没有 draft model
# 没有 EAGLE 权重加载逻辑</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">class EagleDraftModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", 0),
        )
        self.fc = nn.Linear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=getattr(config, "bias", True),
        )
        self.layers = nn.ModuleList([
            EagleDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(self, features, input_ids, position_offset=0):
        inputs_embeds = self.embed_tokens(input_ids).to(features.dtype)
        x = self.fc(torch.cat((inputs_embeds, features), dim=-1))
        positions = torch.arange(
            position_offset,
            position_offset + input_ids.size(1),
            device=input_ids.device,
        )
        for layer in self.layers:
            x = layer(x, positions)
        return x


def load_eagle_draft(path, target_model, target_config):
    config = _read_config(path, target_config)
    model = EagleDraftModel(config)

    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            target_model.model.embed_tokens.weight.detach().cpu()
        )

    state = _load_state_dict(path)
    model.load_state_dict(state, strict=False)
    return model</code></pre>
  </div>
</div>

<div class="note">真实部署时，`path` 应该指向训练好的 EAGLE draft 权重目录。只有 config 的 dummy draft 只能验证流程，不能代表真实性能。</div>

---

## 5. ModelRunner 初始化：加载 draft model

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">torch.set_default_device("cuda")
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)
self.sampler = Sampler()
self.warmup_model()
self.allocate_kv_cache()

if not self.enforce_eager:
    self.capture_cudagraph()</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">torch.set_default_device("cuda")
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)

if getattr(config, "quant_mode", "none") == "dynamic":
    from nanovllm.quant_utils import apply_dynamic_int8
    self.model = apply_dynamic_int8(self.model)

self.eagle_draft = None
if config.eagle_model is not None:
    assert self.world_size == 1, \
        "EAGLE v1 path currently supports tensor_parallel_size=1"
    self.eagle_draft = load_eagle_draft(
        config.eagle_model,
        self.model,
        hf_config,
    )
    self.eagle_draft.to(
        device="cuda",
        dtype=hf_config.torch_dtype,
    ).eval()

self.sampler = Sampler()
self.warmup_model()
self.allocate_kv_cache()

if not self.enforce_eager:
    self.capture_cudagraph()</code></pre>
  </div>
</div>

---

## 6. ModelRunner：普通 decode vs EAGLE decode

原始 nano-vLLM 每轮 decode 只生成一个 token。EAGLE 路径一轮可能接受多个 token。

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">def run(self, seqs, is_prefill):
    input_ids, positions = (
        self.prepare_prefill(seqs)
        if is_prefill
        else self.prepare_decode(seqs)
    )
    temperatures = self.prepare_sample(seqs)
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist()
    reset_context()
    return token_ids</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">def run_eagle(self, seqs):
    assert len(seqs) == 1
    seq = seqs[0]
    if self.eagle_draft is None or seq.temperature &gt; 1e-10:
        return self.run(seqs, False)

    # 1. target model 先看当前 last token
    input_ids, positions = self.prepare_decode(seqs)
    logits, features = self.run_model_with_features(input_ids, positions)
    seed_token = int(torch.argmax(logits[-1], dim=-1).item())
    seed_feature = features[-1]
    reset_context()

    # 2. draft model 生成多个 candidate token
    candidates = self.generate_eagle_candidates(
        seq,
        seed_token,
        seed_feature,
    )

    # 3. target model 一次性验证 candidate tokens
    verify_ids = [seq.last_token] + candidates
    input_ids, positions = self.prepare_extend(seq, verify_ids)
    logits, features = self.run_model_with_features(input_ids, positions)
    reset_context()

    # 4. 从左到右接受 token
    accepted = []
    for i, candidate in enumerate(candidates):
        target_token = int(torch.argmax(logits[i], dim=-1).item())
        if target_token == candidate:
            accepted.append(candidate)
        else:
            accepted.append(target_token)
            return accepted

    accepted.append(int(torch.argmax(logits[len(candidates)], dim=-1).item()))
    return accepted</code></pre>
  </div>
</div>

---

## 7. Scheduler：一次提交多个 accepted token

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">def schedule(self):
    ...
    # decode
    while self.running and num_seqs &lt; self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False


def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if should_finish(seq, token_id):
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">def schedule(self, lookahead_slots: int = 1):
    ...
    # decode
    while self.running and num_seqs &lt; self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append_slots(seq, lookahead_slots):
            if self.running:
                self.preempt(self.running.pop())
            else:
                self.preempt(seq)
                break
        else:
            num_seqs += 1
            if lookahead_slots == 1:
                self.block_manager.may_append(seq)
            else:
                self.block_manager.ensure_append_slots(seq, lookahead_slots)
            scheduled_seqs.append(seq)

    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False


def postprocess_many(self, seq, token_ids):
    old_len = len(seq)
    for token_id in token_ids:
        if seq.num_completion_tokens &gt;= seq.max_tokens:
            break
        seq.append_token(token_id)
        if not seq.ignore_eos and token_id == self.eos:
            break

    self.block_manager.trim_unused_blocks(seq)
    self.block_manager.commit_appended_tokens(seq, old_len)

    if should_finish(seq):
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)</code></pre>
  </div>
</div>

---

## 8. BlockManager：预留并回收 lookahead KV block

EAGLE 一轮可能预留多个 token 的 KV 空间，但最后只接受其中一部分。没用上的 block 必须回收，否则下一轮会把错误 block 当成当前序列的一部分。

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">def can_append(self, seq):
    return len(self.free_block_ids) &gt;= (
        len(seq) % self.block_size == 1
    )


def may_append(self, seq):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]

    if len(seq) % self.block_size == 1:
        assert last_block.hash != -1
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)

    elif len(seq) % self.block_size == 0:
        assert last_block.hash == -1
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[block_table[-2]].hash \
            if len(block_table) &gt; 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">def can_append_slots(self, seq, num_slots):
    if num_slots &lt;= 0:
        return True
    covered_tokens = len(seq) + num_slots - 1
    required_blocks = (
        covered_tokens + self.block_size - 1
    ) // self.block_size
    return len(self.free_block_ids) &gt;= max(
        0,
        required_blocks - len(seq.block_table),
    )


def ensure_append_slots(self, seq, num_slots):
    if num_slots &lt;= 0:
        return
    covered_tokens = len(seq) + num_slots - 1
    required_blocks = (
        covered_tokens + self.block_size - 1
    ) // self.block_size
    while len(seq.block_table) &lt; required_blocks:
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        seq.block_table.append(block_id)


def trim_unused_blocks(self, seq):
    while len(seq.block_table) &gt; seq.num_blocks:
        block_id = seq.block_table.pop()
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)</code></pre>
  </div>
</div>

---

## 9. LLMEngine：判断是否进入 EAGLE 路径

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 nano-vLLM</div>
    <pre><code class="language-python">def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)

    outputs = [
        (seq.seq_id, seq.completion_token_ids)
        for seq in seqs
        if seq.is_finished
    ]
    num_tokens = (
        sum(len(seq) for seq in seqs)
        if is_prefill
        else -len(seqs)
    )
    return outputs, num_tokens</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：接入 EAGLE 后</div>
    <pre><code class="language-python">def step(self):
    can_try_eagle = (
        self.config.eagle_model is not None
        and not self.scheduler.waiting
        and len(self.scheduler.running) == 1
    )
    lookahead_slots = (
        self.config.eagle_max_draft_tokens + 1
        if can_try_eagle
        else 1
    )
    seqs, is_prefill = self.scheduler.schedule(lookahead_slots)

    if (
        self.config.eagle_model is not None
        and not is_prefill
        and len(seqs) == 1
        and seqs[0].temperature &lt;= 1e-10
    ):
        token_ids = self.model_runner.call("run_eagle", seqs)
        self.scheduler.postprocess_many(seqs[0], token_ids)
    else:
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

    outputs = [
        (seq.seq_id, seq.completion_token_ids)
        for seq in seqs
        if seq.is_finished
    ]
    num_tokens = (
        sum(len(seq) for seq in seqs)
        if is_prefill
        else -len(token_ids)
    )
    return outputs, num_tokens</code></pre>
  </div>
</div>

---

## 10. 对比脚本：vanilla vs EAGLE

<div class="compare-wrap">
  <div class="compare-card">
    <div class="compare-title">左栏：原始 bench.py</div>
    <pre><code class="language-python">path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(path, enforce_eager=False, max_model_len=4096)

sampling_params = [
    SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=randint(100, max_output_len),
    )
    for _ in range(num_seqs)
]

llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)</code></pre>
  </div>
  <div class="compare-card">
    <div class="compare-title">右栏：新增 bench_eagle.py</div>
    <pre><code class="language-python">def run_case(name, model_path, prompts, max_tokens, eagle_model):
    kwargs = dict(
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.98,
    )
    if eagle_model is not None:
        kwargs.update(
            eagle_model=eagle_model,
            eagle_max_draft_tokens=4,
        )

    llm = LLM(model_path, **kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    llm.exit()
    return outputs</code></pre>
  </div>
</div>

运行：

```bash
cd /home/legion/nano-vllm
~/vllm_deploy/.venv/bin/python bench_eagle.py \
  --max-tokens 64 \
  --prompt 'hello' \
  --eagle-model /path/to/eagle_draft_model
```

输出示例：

```text
vanilla: 4 tokens, 4.92s, 0.81 tok/s
eagle: 4 tokens, 2.34s, 1.71 tok/s
exact token match: 1/1
```

---

## 11. Smoke Test

静态检查：

```bash
cd /home/legion/nano-vllm
~/vllm_deploy/.venv/bin/python -m py_compile \
  nanovllm/config.py \
  nanovllm/sampling_params.py \
  nanovllm/layers/sampler.py \
  nanovllm/engine/block_manager.py \
  nanovllm/engine/scheduler.py \
  nanovllm/engine/sequence.py \
  nanovllm/models/qwen3.py \
  nanovllm/eagle.py \
  nanovllm/engine/model_runner.py \
  nanovllm/engine/llm_engine.py \
  bench_eagle.py
```

vanilla 路径：

```bash
~/vllm_deploy/.venv/bin/python bench_eagle.py \
  --max-tokens 1 \
  --prompt 'hello'
```

EAGLE 路径：

```bash
~/vllm_deploy/.venv/bin/python bench_eagle.py \
  --max-tokens 4 \
  --prompt 'hello' \
  --eagle-model /path/to/eagle_draft_model
```

---

## 12. 总结

这次部署 EAGLE，真正需要改的是推理状态流：

1. `Config` 增加 EAGLE 开关。
2. `Sampler` 支持 greedy。
3. `Qwen3Model` 返回中间 feature。
4. 新增 EAGLE draft 模型。
5. `ModelRunner` 增加 draft + verify。
6. `Scheduler` 支持一次提交多个 token。
7. `BlockManager` 支持 lookahead KV block，并回收未使用 block。
8. `bench_eagle.py` 做 vanilla 和 EAGLE 的一致性、吞吐对比。

最重要的一点：EAGLE 路径必须是显式启用、可回退的。这样 nano-vLLM 原有代码仍然能正常工作，而我们可以在单序列 greedy 场景下逐步验证 speculative decoding 的正确性和收益。
