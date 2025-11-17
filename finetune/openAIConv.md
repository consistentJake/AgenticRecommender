help me find a open source script or you write one, for using LLaMA-Factory to do LORA finetuning on qwen 3 - 8B model on 4 bit quantization, using rank as 16. I want to use moivelens dataset for a sequential recommendation, which is the last 15 historical movie view history, to predict yes or no for a new movie recommendaiton. help me construct the data clean up, the training data format design, the qwen chat format based data inputs. and also testing module. like before finetune the modle return x result, after finetune, it return y results.

---

Here’s a full pipeline you can actually drop into your repo and tweak: data → alpaca-style JSON → LLaMA-Factory LoRA config for **Qwen3-8B** with **4-bit QLoRA** and **rank=16** → simple before/after test script.

I’ll assume:

- Base: `Qwen/Qwen3-8B-Instruct` (or `Qwen3-8B` if you prefer base).
- Dataset: **MovieLens “latest-small”** (`ratings.csv`, `movies.csv`).
- Framework: **LLaMA-Factory** with `template: qwen3`, 4-bit QLoRA via `quantization_bit: 4`, `quantization_method: bnb` (as in public Qwen3 + LLaMA-Factory examples).
- Task: given last **15 watched movies** → predict **Yes/No** for recommending a candidate movie.

---

## 1\. Data cleanup & sequence construction (MovieLens → seq rec)

MovieLens `ratings.csv` looks like:

Sorted by `userId` and `timestamp` (or `movieId` in some versions).  
`movies.csv` has `movieId,title,genres`.

We’ll make training tuples:

- For each `userId`:
	- Sort interactions by `timestamp`.
	- Keep only ratings with **rating ≥ 3.5 or 4.0** as “liked” (you pick threshold, below I use `>= 4.0` = Yes).
	- For each position `t >= 15`:
		- history = last 15 `movieId` s before t
		- candidate = movie at position `t`
		- label = `"Yes"` if `rating >= 4.0` else `"No"`

We’ll also split into train/val/test (e.g., last 10% per user as test).

### prepare\_movielens\_qwen3.py

This script:

1. Reads MovieLens.
2. Builds sequences.
3. Writes **two JSON files** in **alpaca** format that LLaMA-Factory expects.
```python
python# prepare_movielens_qwen3.py
import json
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

DATA_DIR = Path("LLaMA-Factory/data")  # adjust if needed
RATING_THRESHOLD = 4.0
HISTORY_LEN = 15

def load_movielens(ml_dir: Path):
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    movies = pd.read_csv(ml_dir / "movies.csv")
    movie_map = dict(zip(movies["movieId"], movies["title"]))
    return ratings, movie_map

def build_sequences(ratings, movie_map):
    # Group by user, sort by timestamp
    ratings = ratings.sort_values(["userId", "timestamp"])
    user_groups = defaultdict(list)

    for row in ratings.itertuples(index=False):
        # row: userId, movieId, rating, timestamp
        user_groups[row.userId].append(
            (row.movieId, float(row.rating), int(row.timestamp))
        )

    examples = []
    for user, events in user_groups.items():
        if len(events) <= HISTORY_LEN:
            continue
        # simple temporal split: last 10% as test
        split_idx = int(len(events) * 0.9)
        for idx in range(HISTORY_LEN, len(events)):
            hist = events[idx - HISTORY_LEN : idx]
            cand_movie, rating, ts = events[idx]

            history_titles = []
            for m_id, r, _ in hist:
                title = movie_map.get(m_id, f"Movie {m_id}")
                history_titles.append(f"{title} (rating ~{r:.1f})")

            cand_title = movie_map.get(cand_movie, f"Movie {cand_movie}")
            label = "Yes" if rating >= RATING_THRESHOLD else "No"

            sample = {
                "user": user,
                "is_test": idx >= split_idx,
                "history_titles": history_titles,
                "candidate_title": cand_title,
                "label": label,
            }
            examples.append(sample)

    return examples

def to_alpaca(example):
    history_str = "; ".join(example["history_titles"])
    instr = (
        "You are a movie recommendation assistant. "
        "Given a user's recent watch history and a candidate movie, "
        "answer with exactly one word: Yes or No. "
        "'Yes' means you would recommend the movie based on the history; "
        "'No' means you would not."
    )
    # Qwen3 chat-style instruction text
    input_text = (
        f"User's last {HISTORY_LEN} watched movies:\n"
        f"{history_str}\n\n"
        f"Candidate movie:\n{example['candidate_title']}\n\n"
        "Should we recommend this movie to the user? Answer Yes or No."
    )
    output_text = example["label"]

    return {
        "instruction": instr,
        "input": input_text,
        "output": output_text,
        "system": "",    # optional
        "history": [],   # no previous rounds; could embed longer convos if you want
    }

def main():
    ml_dir = Path("path/to/ml-latest-small")  # TODO: change
    ratings, movie_map = load_movielens(ml_dir)
    examples = build_sequences(ratings, movie_map)
    random.shuffle(examples)

    train_data = [to_alpaca(e) for e in examples if not e["is_test"]]
    test_data  = [to_alpaca(e) for e in examples if e["is_test"]]

    out_train = DATA_DIR / "movielens_qwen3_seqrec_train.json"
    out_test  = DATA_DIR / "movielens_qwen3_seqrec_test.json"

    out_train.write_text(json.dumps(train_data, ensure_ascii=False, indent=2))
    out_test.write_text(json.dumps(test_data, ensure_ascii=False, indent=2))
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

if __name__ == "__main__":
    main()
```

---

## 2\. LLaMA-Factory dataset config (alpaca + Qwen3 template)

LLaMA-Factory wants your custom dataset registered in `data/dataset_info.json`, in **alpaca format** like:

### Sample training item (for sanity check)

After running the prep script, a single entry will look roughly like:

With `template: qwen3`, LLaMA-Factory will apply Qwen3’s built-in chat template (using the `chat_template.jinja` shipped with the model) and wrap this as system/user/assistant messages under the hood.

---

## 3\. LoRA + 4-bit QLoRA config for Qwen3-8B in LLaMA-Factory

There are public examples of Qwen3 SFT via `llamafactory-cli train` using `template: qwen3`, LoRA, and 4-bit quant: `quantization_bit: 4`, `quantization_method: bnb`, `double_quantization: True`.

Here’s a YAML config tuned for **Qwen/Qwen3-8B-Instruct**, **LoRA rank=16**, **4-bit QLoRA** on a **single 4090**.

### examples/train/movielens\_qwen3\_lora.yaml

```yaml
yaml### model
model_name_or_path: Qwen/Qwen3-8B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### quantization (4-bit QLoRA)
quantization_bit: 4
quantization_method: bnb      # bitsandbytes 4bit
double_quantization: true
# you can also add:
# quantization_config:
#   bnb_4bit_compute_dtype: bfloat16
#   bnb_4bit_use_double_quant: true
#   bnb_4bit_quant_type: nf4

### dataset
dataset_dir: LLaMA-Factory/data
dataset: movielens_qwen3_seqrec
template: qwen3
cutoff_len: 1024
max_samples: 200000        # cap if dataset is huge
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 2

### training
output_dir: saves/Qwen3-8B-Instruct/movielens_lora
num_train_epochs: 3.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_steps: 200
max_grad_norm: 1.0
optim: adamw_torch

logging_steps: 10
save_steps: 500
plot_loss: true
packing: false
ddp_timeout: 180000000

bf16: true                 # if your GPU supports bf16
flash_attn: auto
include_num_input_tokens_seen: true

report_to: none            # or "wandb", "tensorboard" if you want
```

Then run:

This is conceptually the same pattern as the public Qwen3 + LLaMA-Factory examples (just with your dataset + LoRA rank updated to 16).

---

## 4\. Testing module: before vs after finetune

We’ll do a **simple offline eval**:

- Use `movielens_qwen3_seqrec_test.json` as your held-out set.
- For each item:
	- Build the same text as in training (`instruction` + `input`).
	- Ask the model to answer briefly.
	- Parse the first token as `"yes"` or `"no"`.
- Compare accuracy:
	- Base = `Qwen/Qwen3-8B-Instruct`
	- Finetuned = base + LoRA adapter saved by LLaMA-Factory.

You *can* also use `llamafactory-cli eval` with `template: qwen3`, but here’s a custom script so you see raw behavior.

### eval\_movielens\_qwen3\_before\_after.py

```python
pythonimport json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 8

BASE_MODEL = "Qwen/Qwen3-8B-Instruct"
ADAPTER_PATH = "saves/Qwen3-8B-Instruct/movielens_lora"  # output_dir above
TEST_JSON = "LLaMA-Factory/data/movielens_qwen3_seqrec_test.json"

def normalize_yn(text: str):
    text = text.strip().lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    # fallback heuristic
    if "yes" in text:
        return "Yes"
    if "no" in text:
        return "No"
    return None

def load_data(n_samples=None):
    data = json.loads(Path(TEST_JSON).read_text())
    if n_samples:
        data = data[:n_samples]
    return data

def build_prompt(item):
    # We just reuse instruction + input; the model's chat_template will wrap it.
    # For Qwen3 Instruct models, the tokenizer.apply_chat_template uses messages.
    instruction = item["instruction"]
    user_input = item["input"]

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"{instruction}\n\n{user_input}"},
    ]
    return messages

def eval_model(model, tokenizer, data, max_samples=None):
    model.eval()
    correct = 0
    total = 0

    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break

        gold = item["output"]  # "Yes" or "No"
        messages = build_prompt(item)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = normalize_yn(gen_text)

        if pred is not None:
            total += 1
            if pred == gold:
                correct += 1

    acc = correct / total if total > 0 else 0.0
    return acc, total

def main():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    data = load_data()

    base_acc, base_n = eval_model(base_model, tokenizer, data, max_samples=1000)
    print(f"Base model accuracy: {base_acc:.3f} on {base_n} samples")

    print("Loading LoRA adapter...")
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    lora_model.to(DEVICE)

    lora_acc, lora_n = eval_model(lora_model, tokenizer, data, max_samples=1000)
    print(f"Finetuned model accuracy: {lora_acc:.3f} on {lora_n} samples")

if __name__ == "__main__":
    main()
```

Sample expected output (numbers are hypothetical):

That gives you exactly the “before finetune X results, after finetune Y results” story.

---

## 5\. A few practical tips

- **Negatives vs positives**: MovieLens is skewed to positives; you may want to:
	- treat `< 3.5` as **No**, and **drop mid ratings** (e.g., 3.0) to reduce noise;
	- or sub-sample negatives so Yes/No are roughly balanced.
- **Context richness**: you can add genres or approximate rating buckets in the history string to give the model more signals (already lightly included as “rating ~4.0”).
- **Hyperparams to watch** (for 4090 + 8B with 4-bit):
	- `per_device_train_batch_size`, `gradient_accumulation_steps`, `cutoff_len` → main VRAM knobs.
	- `learning_rate` (1e-4 vs 5e-5) and `num_train_epochs` → over/under-fitting.
	- `lora_rank` (you already set 16) and `lora_target` (`all` vs specific projections) if you see underfitting.

If you want, next step I can help you:

- switch to ShareGPT-style multi-turn data (e.g., “User: I liked A/B/C, what else?”),
- or add proper metrics via `llamafactory-cli eval` so everything stays inside the same framework.

---

can you research online, can you not using llamafactory-cli, we can do everything in python and even notebooks so everything can be visualized?

---

Yep, we can absolutely ditch `llamafactory-cli` and do everything in **pure Python / notebooks**, while still getting:

- Qwen3-8B-Instruct
- 4-bit QLoRA
- LoRA rank = 16
- MovieLens sequential rec → **Yes/No** chat outputs
- Before vs after finetune evaluation

Below is a **notebook-style pipeline** you can paste into Jupyter (or turn into `.py` scripts). I’ll keep it linear and runnable.

---

## 0\. Install & imports (one-time)

Then in your notebook:

We’ll follow the **SFTTrainer + PEFT LoRA + 4-bit QLoRA** pattern used in recent Qwen3 tutorials and TRL docs.

---

## 1\. Prepare MovieLens sequential data (last 15 → Yes/No)

We’ll use **MovieLens latest-small** (ratings + movies).

Assumptions:

- Data at `path/to/ml-latest-small/ratings.csv` & `movies.csv`
- Label rule:
	- rating ≥ 4.0 → `"Yes"`
	- rating < 4.0 → `"No"`
```python
python# cell 2: movielens utilities

ML_PATH = Path("path/to/ml-latest-small")  # TODO: change
RATING_THRESHOLD = 4.0
HISTORY_LEN = 15

def load_movielens(ml_dir: Path):
    ratings = pd.read_csv(ml_dir / "ratings.csv")
    movies = pd.read_csv(ml_dir / "movies.csv")
    movie_title = dict(zip(movies["movieId"], movies["title"]))
    return ratings, movie_title

def build_seq_examples(ratings: pd.DataFrame, movie_title: dict):
    # sort by user, then timestamp
    ratings = ratings.sort_values(["userId", "timestamp"])
    user_hist = defaultdict(list)

    for row in ratings.itertuples(index=False):
        user_hist[row.userId].append(
            (row.movieId, float(row.rating), int(row.timestamp))
        )

    examples = []
    for user, events in user_hist.items():
        if len(events) <= HISTORY_LEN:
            continue

        # simple temporal split: last 10% indices = test
        split_idx = int(len(events) * 0.9)

        for t in range(HISTORY_LEN, len(events)):
            hist = events[t - HISTORY_LEN : t]
            cand_movie, rating, ts = events[t]

            history_titles = [
                f"{movie_title.get(m_id, f'Movie {m_id}')} (rating ~{r:.1f})"
                for m_id, r, _ in hist
            ]
            candidate_title = movie_title.get(cand_movie, f"Movie {cand_movie}")

            label = "Yes" if rating >= RATING_THRESHOLD else "No"

            examples.append(
                {
                    "user_id": user,
                    "is_test": t >= split_idx,
                    "history_titles": history_titles,
                    "candidate_title": candidate_title,
                    "label": label,
                }
            )
    return examples

ratings_df, movie_title_map = load_movielens(ML_PATH)
raw_examples = build_seq_examples(ratings_df, movie_title_map)
len(raw_examples)
```

You can inspect a few:

---

## 2\. Turn it into chat-style examples for Qwen3

We’ll keep the model in “instruction-following chat mode” and constrain the output to **exactly “Yes” or “No”**.

We’ll store each sample as:

- `messages` = list of `{role, content}` for use with Qwen3’s `chat_template`.
```python
python# cell 3: build HF datasets with chat messages

def make_messages(example):
    history_str = "; ".join(example["history_titles"])
    sys_prompt = (
        "You are a movie recommendation assistant. "
        "Given a user's recent watch history and a candidate movie, "
        "you must answer with exactly one word: Yes or No.\n"
        "- Yes: you recommend the movie.\n"
        "- No: you do not recommend the movie.\n"
        "Do not include any extra words."
    )
    user_content = (
        f"User's last {HISTORY_LEN} watched movies:\n"
        f"{history_str}\n\n"
        f"Candidate movie:\n{example['candidate_title']}\n\n"
        "Should we recommend this movie to the user? Answer Yes or No."
    )
    assistant_content = example["label"]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return messages

for ex in raw_examples[:3]:
    ex["messages"] = make_messages(ex)

# split into train / test
train_rows = [e for e in raw_examples if not e["is_test"]]
test_rows  = [e for e in raw_examples if e["is_test"]]

train_ds = Dataset.from_list(
    [{"messages": e["messages"], "label": e["label"]} for e in train_rows]
)
test_ds = Dataset.from_list(
    [{"messages": e["messages"], "label": e["label"]} for e in test_rows]
)

datasets = DatasetDict({"train": train_ds, "test": test_ds})
datasets
```

---

## 3\. Load Qwen3-8B-Instruct in 4-bit QLoRA

We follow typical QLoRA patterns with **bitsandbytes 4-bit** + **PEFT LoRA**, as in common tutorials.

Define LoRA config (rank=16):

(There are LocalLLaMA posts showing Qwen3 LoRA ranks & behavior; rank 8–16 usually preserves the model’s original capabilities reasonably well. )

---

## 4\. Turn chat messages → final training text

We’ll pre-render each sample into a single string using the **Qwen3 chat template** via `tokenizer.apply_chat_template`. Then `SFTTrainer` just trains on `text` field.

Quick sanity check:

You should see something like `<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant Yes`.

---

## 5\. Train with SFTTrainer (LoRA + 4-bit)

We now plug everything into TRL’s `SFTTrainer`. This pattern is also used in official docs and blog posts for Qwen3 fine-tuning.

At this point you have a 4-bit base model + LoRA adapter checkpoints saved to `OUTPUT_DIR`.

---

## 6\. Evaluation: before vs after finetune (Yes/No accuracy)

Now let’s build a simple evaluation function:

- For each sample in the **test set**:
	- Build messages = system + user, **without** the assistant turn.
	- Use `apply_chat_template(..., add_generation_prompt=True)` to append assistant prompt.
	- Generate a short answer, map to Yes / No.
	- Compare with gold label.

### 6.1 Utility functions

```python
python# cell 8: eval utilities

MAX_NEW_TOKENS = 8

def build_eval_messages(example):
    """System + user only (no assistant) for prompting."""
    messages = example["messages"][:-1]  # drop assistant from original
    return messages

def normalize_yn(text: str):
    t = text.strip().lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    if "yes" in t:
        return "Yes"
    if "no" in t:
        return "No"
    return None

def eval_model(base_model, tokenizer, raw_examples, max_samples=500):
    base_model.eval()
    device = next(base_model.parameters()).device

    total = 0
    correct = 0

    for i, ex in enumerate(raw_examples):
        if i >= max_samples:
            break

        gold = ex["label"]
        messages = build_eval_messages(ex)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = base_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred = normalize_yn(gen_text)

        if pred is None:
            continue

        total += 1
        if pred == gold:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return acc, total
```

### 6.2 Evaluate base vs finetuned

```python
python# cell 9: evaluate

# 1) evaluate **base model** (already loaded as \`model\` before training, so reload fresh)
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

base_model_fp16 = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=DEVICE_MAP,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)
base_model_fp16.config.use_cache = True

base_acc, base_n = eval_model(
    base_model_fp16,
    base_tokenizer,
    test_rows,
    max_samples=500,
)
print(f"Base model accuracy: {base_acc:.3f} on {base_n} samples")

# 2) load **finetuned LoRA** model
ft_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
if ft_tokenizer.pad_token is None:
    ft_tokenizer.pad_token = ft_tokenizer.eos_token

ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map=DEVICE_MAP,
    trust_remote_code=True,
)
from peft import PeftModel
ft_model = PeftModel.from_pretrained(ft_model, OUTPUT_DIR)
ft_model.config.use_cache = True

ft_acc, ft_n = eval_model(
    ft_model,
    ft_tokenizer,
    test_rows,
    max_samples=500,
)
print(f"Finetuned model accuracy: {ft_acc:.3f} on {ft_n} samples")
```

You’ll see something like:

This gives you the **“before finetune X results, after finetune Y results”** story you wanted, entirely in Python.

---

## 7\. Where this matches “online” patterns

What we just built is essentially the same architecture you see in:

- **TRT/TRL SFTTrainer docs** for fine-tuning Qwen3 chat models.
- **QLoRA + PEFT tutorials** that use BitsAndBytes 4-bit and LoRAConfig.
- Recent **blog posts on fine-tuning Qwen3 with LoRA** using Python-only workflows (Unsloth / TRL).

The main difference is we’re:

- using **MovieLens sequential rec** as our dataset,
- forcing the model into a **strict Yes/No classification** space via the prompt and label,
- doing everything in a **notebook-friendly, fully visible** way—no CLI wrappers.

---

If you’d like, next step I can:

- add **TensorBoard / WandB logging** cells so you can visualize loss over steps;
- or tweak the **data design** (e.g., including genres, timestamps, or “user likes sci-fi” summaries) to see if it boosts accuracy.