Short answer: your setup looks solid overall, and **your raw training speed (2.2–2.3 it/s) is *normal*** for a 0.6B model with QLoRA + gradient checkpointing on a 3090. The *real* problem is that **evaluation is dominating your wall-clock time**, so the *effective* training speed is much slower than you think.

I’ll break this into two parts:

---

## 1. Is the training speed reasonable?

### 1.1. Forward/backward speed

You reported:

* **2.0–2.4 it/s** training
* **~0.43–0.50 seconds / step**

Given:

* Qwen3-0.6B (0.6B params) ([GitHub][1])
* QLoRA (4-bit) + gradient checkpointing + BF16 compute
* Context length up to 2048
* Single RTX 3090, 24 GB VRAM

Community benchmarks for *training* (not just inference) on a 3090 with:

* **7B models** + QLoRA + seq_len=512 report “a few thousand tokens/sec” depending on batch and optimizations. ([DeepResearch.wiki][2])

You are on:

* **Much smaller model (0.6B)** → cheaper per forward
* But **SFT training (forward + backward)** + **gradient checkpointing (~20% slowdown)** + **longer sequences (up to 2048)**

If your average sequence length is even ~512–1024 tokens, your effective throughput is roughly:

* 2.2 it/s × 512 tokens ≈ **~1100 tokens/sec** (ballpark)

That’s entirely consistent with “few thousand tokens/sec for a 7B model at shorter length” from the literature. ([DeepResearch.wiki][2])

👉 **Conclusion:**
**Your *per-step* training speed is OK.** It’s not suspiciously slow for this setup.

---

### 1.2. Where time is really going: evaluation

Your eval config:

* **Eval dataset:** 4,450 samples
* **Per-device eval batch size:** 1
* **Eval speed:** 7.86–8.82 steps/s
* **Eval runtime:** 504–565 sec → **~8–9 minutes per eval**
* **eval_steps = 200** → you evaluate *very* often

Let’s look at one “cycle” of 200 steps:

* **Training 200 steps:**

  * 200 steps ÷ 2.3 it/s ≈ **87 s (~1.5 min)**
* **Evaluation (4,450 steps @ ~8 steps/s):**

  * 4,450 ÷ 8 ≈ **556 s (~9.3 min)**

So every 200 training steps costs:

> **~1.5 min train + ~9 min eval ≈ 10.5 min total**

Effective steps/hour:

* 200 steps / (10.5 min ÷ 60) ≈ **~1,115 steps/hour** (I did the exact math)

Total planned steps: **234,813**

* 234,813 ÷ 1,115 ≈ **~211 hours ≈ 8.8 days**

👉 This is the key:
Your **27–33 hour estimate only accounts for training forward/backward**, *not* the 8–9 minute evaluations every 200 steps.

If you keep this config, the full 3-epoch run is closer to **~9 days**, not ~1–1.5 days.

So:

* **Per-step speed:** fine
* **End-to-end wall-clock speed:** effectively *very slow* because eval is eating the time

---

## 2. Suggestions to improve (from “internet experience”)

Here are the things I’d change, roughly in order of impact.

### 2.1. Fix evaluation: this is your biggest win

Right now eval is the main bottleneck.

**Suggestions:**

1. **Increase `per_device_eval_batch_size`**

   * You use only ~16–18 GB VRAM out of 24 GB for training. At eval time you don’t need gradients or checkpointing, so memory usage is much lower.
   * Try **eval batch size = 8, 16, or even 32** and see what fits.
   * If you can go from batch 1 → 16, your eval time could drop from **9 minutes → ~30–40 seconds** per evaluation.

2. **Evaluate less frequently**

   * `evaluation_strategy="steps"` with `eval_steps=200` is overkill for 234k total steps.
   * Common patterns in SFT guides: eval **once per epoch**, or every **1–5k steps** for longer runs. ([Medium][3])
   * Good compromise for your run:

     * `eval_steps = 2000` or `eval_steps = 5000`
   * That alone gives you ~10–25× fewer evals.

3. **Use a smaller eval subset for frequent checks**

   * Keep full 4,450 samples for *final* evaluation.
   * For training monitoring, create e.g. **“movielens_qwen3_eval_small”** with 500–1000 rows.
   * Many practical QLoRA guides recommend small validation sets during long runs to keep iteration time reasonable. ([Medium][3])

If you do all three (larger eval batch, less frequent eval, smaller eval split for frequent checks), you can easily move from **"9 days" territory down to < 1 day** for the same 3 epochs.

---

### 2.2. Adjust effective batch size (stability & convergence)

You’re currently at:

* `per_device_train_batch_size = 1`
* `gradient_accumulation_steps = 1`
* **Effective batch size = 1**

That’s extremely small. Most LoRA/QLoRA examples for small/mid models recommend effective batch sizes in the **16–64** range for more stable gradients. ([Medium][4])

On a 3090 with 4-bit QLoRA on a 0.6B model, you almost certainly can afford more.

Two options:

1. **Increase per-device batch size**

   * Try `per_device_train_batch_size = 4` or `8`. Let OOM be your guide.
   * Keep `gradient_accumulation_steps = 1`.

2. **Use gradient accumulation to simulate a larger batch**

   * Example:

     * `per_device_train_batch_size = 2`
     * `gradient_accumulation_steps = 8`
     * → effective batch size = 16
   * This is explicitly recommended in LoRA/QLoRA best-practice articles. ([Medium][4])

**Why it helps:**

* Smoother updates → better convergence, often needs fewer steps for similar performance.
* You might be able to **reduce total epochs** once you see the loss curve stabilizing.

---

### 2.3. Rethink sequence length & checkpointing

You configured:

* `max_seq_length` (context length) = 2048
* **Gradient checkpointing: enabled**

These are good defaults for safety, but maybe conservative for your task.

**Questions to ask yourself:**

1. **What is the *real* average sequence length in your SFT examples?**

   * If most samples are < 512–1024 tokens, training at 2048 is wasted compute.
   * Trimming to a tighter max length (e.g., 1024) can give **~2× speedup** in FLOPs per step.

2. **Do you actually *need* gradient checkpointing for this model on 24GB?**

   * 0.6B + QLoRA is relatively light. Many 7B QLoRA setups run on 24GB without checkpointing. ([Runpod][5])
   * If disabling it fits VRAM:

     * You may gain ~20–30% speed.

So a good plan:

* **Measure max and p95 sequence lengths** in your dataset.
* Set **max length = min(2048, p99)**.
* Try a run with **gradient checkpointing disabled** and check VRAM with `nvidia-smi`.

---

### 2.4. Training length, LR, and monitoring

Some comments on your hyperparameters:

* **LR = 1.5e-4** with LoRA/QLoRA on small models is totally reasonable.
  Many guides suggest `1e-4` to `2e-4` as the sweet spot. ([Medium][4])
* **Warmup ratio = 0.03** is fine; you could shorten it (since 7k warmup steps is a lot), but it’s not a big problem.

More important:

1. **Do you really need 3 full epochs?**

   * You have **78,271 training examples** → 234,813 steps at batch 1.
   * For binary recommendation SFT, you might find that performance saturates earlier (e.g., after 1–2 epochs).
   * Track eval loss/metrics per checkpoint and consider **early stopping** once improvements flatten or start to reverse.

2. **Track more than loss**

   * For a **binary movie recommendation** task, add:

     * Accuracy
     * AUC
     * F1
   * Helps you see if loss wiggles (0.37–0.61 for now) actually correspond to meaningful performance changes.

---

### 2.5. Minor LoRA config thoughts

Your LoRA setup:

* **Rank = 16**, **alpha = 64**, dropout=0.05
* Targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
* ~1.67% trainable params (10M)

This is very standard and perfectly fine. A couple optional knobs:

* If you find underfitting, you *could* try:

  * Rank 32 (at some extra VRAM/compute cost)
* If you see overfitting, you can:

  * Slightly increase dropout to 0.1
  * Reduce epochs, or use early stopping

But I wouldn’t touch these first; the big low-hanging fruit is eval/batch/seq length.

---

## 3. TL;DR recommendations

If you want a concrete “next run” config that’s much more time-friendly, I’d try:

1. **Eval strategy**

   * `eval_steps = 2000` (or per epoch)
   * `per_device_eval_batch_size = 16` (or as high as VRAM allows)
   * Optionally: create a smaller eval split (500–1000 examples) for frequent checks

2. **Batch size**

   * `per_device_train_batch_size = 2–4`
   * `gradient_accumulation_steps = 8` (effective batch 16–32)

3. **Sequence length / GC**

   * Set max length to something close to your p95 sequence length (e.g. 1024 if that’s enough)
   * Try turning **off** gradient checkpointing if VRAM allows

4. **Training length & metrics**

   * Keep LR = 1.5e-4, warmup_ratio = 0.03
   * Add metrics like accuracy/AUC/F1
   * Consider stopping early if eval metrics stop improving

If you’d like, next step I can help you:

* Derive **new time estimates** for a specific modified config (e.g. eval every 2000 steps, eval batch 16), or
* Look at some sample logs (loss vs step) and suggest whether you might be over- or under-training.

[1]: https://raw.githubusercontent.com/mudler/LocalAI/refs/heads/master/gallery/index.yaml?utm_source=chatgpt.com "https://raw.githubusercontent.com/mudler/LocalAI/r..."
[2]: https://www.deepresearch.wiki/report/comprehensive-review-fine-tuning-llm-libraries-veqgqckgr85lchkua3me3jpu?utm_source=chatgpt.com "Comprehensive Review of Fine-Tuning LLM Libraries and ..."
[3]: https://medium.com/%40QuarkAndCode/lora-qlora-llm-fine-tuning-best-practices-setup-pitfalls-c8147d34a6fd?utm_source=chatgpt.com "LoRA & QLoRA LLM Fine‑Tuning: Best Practices, Setup ..."
[4]: https://medium.com/%40raquelhvaz/efficient-llm-fine-tuning-with-lora-e5edb88b64a1?utm_source=chatgpt.com "Efficient LLM Fine-Tuning with LoRA | by Raquel Vaz, PhD"
[5]: https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget?utm_source=chatgpt.com "How can I fine-tune large language models on a budget ..."
