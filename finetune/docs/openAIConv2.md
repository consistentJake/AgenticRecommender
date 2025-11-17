\## Deep Research Prompt (for Qwen3 LoRA + Recommender System) > \*\*Goal\*\* > I want to build a \*practical, open-source\* LoRA fine-tuning setup for a \*\*Qwen3 model\*\* on a \*\*recommendation dataset\*\*, and run it on a \*\*single RTX 4090\*\*. > The recommendation task is: > > \* \*\*Input\*\*: > > \* a user’s \*\*purchase history\*\*: list of product IDs (or titles) > \* a \*\*candidate product\*\* > \* \*\*Output\*\*: > > \* a binary label: \*\*“yes” (recommend) / “no” (do not recommend)\*\* > > I need you to do \*\*deep research\*\* (GitHub, blogs, papers, Reddit, etc.) and then give me a \*\*very concrete blueprint\*\*: repositories, code patterns, hyperparameters, and monitoring strategy. > > --- > > ### 1. Find open-source LoRA examples relevant to Qwen3 + recommendation > > 1. Search \*\*GitHub\*\* and the wider web for: > > \* LoRA / QLoRA / DoRA \*\*fine-tuning examples on Qwen / Qwen2 / Qwen2.5 / Qwen3\*\* > \* LoRA fine-tuning for \*\*recommender systems\*\* or \*\*CTR prediction\*\* or \*\*ranking / binary classification\*\* tasks > \* General LLM-for-recommendation examples that are: > > \* instruction-style or chat-style training > \* using LoRA/PEFT/Unsloth/TRL/DeepSpeed, etc. > 2. For each \*\*relevant repository or code example\*\*, give me: > > \* \*\*Name + link\*\* > \* Which \*\*model family\*\* (Qwen, LLaMA, others) and \*\*size\*\* > \* Which \*\*libraries\*\* are used (Hugging Face Transformers, PEFT, TRL, Unsloth, vLLM, etc.) > \* \*\*Task type\*\* (e.g., recommendation / CTR / binary classification / ranking / instruction-tuning) > \* How the \*\*input/output format\*\* looks in the dataset (especially if similar to: purchase history + candidate item → yes/no) > \* \*\*Training hardware\*\* they used (if mentioned) and whether it is plausible to run on \*\*one RTX 4090 (24 GB)\*\* with careful settings > 3. Prioritize: > > \* Repos that use \*\*Qwen / Qwen3\*\* directly > \* Repos that do \*\*recommendation / ranking / binary classification\*\* with LoRA, even if not on Qwen (I can adapt those patterns) > \* Repos with \*\*clear training scripts\*\*, not just abstract demos > > In your final list, clearly mark: > > \* “\*\*Qwen-native\*\*” examples > \* “\*\*Non-Qwen but highly adaptable to Qwen3\*\*” examples > > --- > > ### 2. Design a data format for my recommendation task > > Based on best practices from the repos and papers you find, propose \*\*2–3 concrete ways\*\* to format my data for Qwen3 + LoRA: > > 1. \*\*Prompt-style format (chat/instruction)\*\* > > \* Example of a single training sample serialized as text, including: > > \* system / user / assistant roles (if relevant) > \* representation of \*\*product history\*\* (IDs vs titles vs titles + categories) > \* representation of the \*\*candidate product\*\* > \* target “yes”/“no” or “relevant”/“not relevant” token(s) > 2. \*\*Classification-style format\*\* > > \* How to phrase the prompt so that the \*\*last token(s)\*\* correspond to a binary label > \* How to constrain the output to a small label space (“yes”/“no”) for more stable training > 3. For each format, discuss: > > \* Pros/cons (e.g., interpretability, label leakage, sequence length, memory usage) > \* How to keep \*\*sequence length\*\* manageable for long purchase histories on a single 4090 > \* Whether to \*\*truncate\*\* or \*\*sample\*\* from a user’s history and how to do that > > Provide \*\*example JSON / JSONL records\*\* and the \*\*exact prompt strings\*\* you recommend. > > --- > > ### 3. Concrete LoRA training plan for Qwen3 on a single RTX 4090 > > Assume: > > \* Model: a reasonably sized \*\*Qwen3 variant\*\* that fits + trains on \*\*one 24 GB RTX 4090\*\* with LoRA > \* Dataset size: between \*\*50k and a few million\*\* labeled pairs > > I want a very \*\*practical, step-by-step setup\*\*, including: > > 1. \*\*Model choice & precision\*\* > > \* Recommend specific \*\*Qwen3 model\*\* and size(s) with reasoning > \* Suggested \*\*precision\*\* (e.g., 4-bit/8-bit base model via bitsandbytes + LoRA in 16-bit) with a concrete memory estimate on 24 GB > 2. \*\*LoRA configuration\*\* > > \* Recommended \*\*target modules\*\* for Qwen3 (e.g., which attention / MLP layers) > \* Suggested defaults and rationales for: > > \* `r` (rank) > \* `alpha` > \* `dropout` > \* whether to use \*\*bias\*\* or not > \* Discuss trade-offs between \*\*smaller r (memory)\*\* vs \*\*larger r (capacity)\*\* for a binary recommendation task > 3. \*\*Training hyperparameters\*\* tuned for a 4090 > > \* Batch size (tokens and samples) and \*\*gradient accumulation steps\*\* for 24 GB > \* Learning rate schedule (e.g., cosine, linear) and \*\*learning rate range\*\* for LoRA adapters > \* Number of \*\*epochs\*\* / max steps, \*\*warmup steps / ratio\*\*, weight decay > \* Recommended \*\*max sequence length\*\* given typical purchase history lengths > \* Mixed precision settings (e.g., bf16 vs fp16) and any stability tips > 4. \*\*Implementation details\*\* > > \* Example training loop design using one of: > > \* raw \*\*Transformers + PEFT\*\*, or > \* \*\*TRL\*\*, or > \* \*\*Unsloth\*\*, if that significantly simplifies Qwen3 fine-tuning > \* Tips to avoid \*\*out-of-memory\*\* on a 4090: > > \* gradient checkpointing > \* `flash_attn` / efficient attention if available > \* smaller context, shorter histories, tokenization tricks > > Provide \*\*pseudocode or a condensed training script outline\*\* that I can turn into real code. It should be \*specific\* to Qwen3, not only generic. > > --- > > ### 4. Key engineering tips and hyperparameters to pay attention to > > Summarize \*\*key engineering tips\*\* specific to: > > 1. \*\*Hyperparameters that really matter\*\* for this use case: > > \* LoRA: `r`, `alpha`, `target_modules`, `dropout` > \* Optimizer choice (e.g., AdamW vs others), learning rate, warmup > \* Sequence length and history truncation strategy > \* Positive/negative sample ratio (e.g., how many “no” samples per “yes”) > 2. \*\*Regularization & generalization\*\* > > \* When to use label smoothing (if at all) > \* Data augmentation strategies (e.g., shuffling history, masking some interactions) > \* Handling user/item cold start if applicable > 3. \*\*Evaluation metrics\*\* > > \* For binary “recommend / not recommend”: > > \* AUC, ROC, accuracy, F1 > \* For “top-K recommendation” framing: > > \* Hit Rate@K, NDCG@K, MRR, etc. > \* How to compute these metrics efficiently on validation sets given large item spaces > 4. \*\*Practical tips from real repos / blog posts\*\* > > \* Any patterns authors mention like: > > \* “small rank + slightly higher LR works better” > \* “freezing more layers to avoid overfitting” > \* “avoid very long histories — hurts convergence” > \* If there are \*\*failure modes\*\* reported (e.g., over-confident “yes”, collapsed predictions), explain them and how to mitigate > > Make this section a \*\*checklist\*\* I can use before starting training. > > --- > > ### 5. Monitoring and debugging during training > > I want a clear plan for \*\*monitoring\*\*, ideally compatible with tools like \*\*Weights & Biases\*\*, TensorBoard, or simple CSV logs. > > 1. \*\*Metrics during training\*\* > > \* What to log every N steps: > > \* training loss > \* validation loss > \* accuracy / F1 > \* AUC for the binary task > \* How to detect: > > \* overfitting (patterns in train vs val curves) > \* underfitting (both losses high, metrics low) > \* label imbalance problems > 2. \*\*Qualitative monitoring\*\* > > \* How to log and inspect \*\*example predictions\*\* over time: > > \* Sample users with their history + candidate items > \* Compare raw model text output (“yes/no”) vs ground truth > \* Track how predictions change across checkpoints > 3. \*\*Resource monitoring\*\* > > \* How to monitor \*\*GPU utilization, memory\*\*, and \*\*tokens per second\*\* > \* What ranges are “healthy” for a LoRA-tuned Qwen3 on one 4090 > \* How to interpret OOM issues and what to shrink first (batch size vs seq length vs model size) > > Please propose a \*\*simple monitoring setup\*\* (e.g., with Weights & Biases or pure Python logging) and describe what graphs I should look at. > > --- > > ### 6. End-to-end recipe & prioritized action list > > Finally, based on all the above research, give me: > > 1. A concise \*\*end-to-end recipe\*\* (bullet-point checklist) for: > > \* preparing the dataset > \* formatting it into prompts > \* configuring Qwen3 + LoRA > \* running training on a single 4090 > \* evaluating and iterating > 2. A \*\*prioritized action list\*\* for me: > > \* Step 1: which repos to clone and study first > \* Step 2: which minimal training script to implement > \* Step 3: which hyperparameters to sweep first > \* Step 4: which monitoring dashboard to set up > > Emphasize \*\*practicality\*\* and \*\*real, runnable code patterns\*\* over abstract theory. If possible, point out \*\*specific files / scripts\*\* in the repos (e.g., `train_lora_qwen.py`, `finetune_rec.py`) that I should read first.

---

Thanks for the detailed prompt! Before I begin, could you please confirm a few quick points so I can tailor everything precisely:

1. Which recommendation dataset(s) are you planning to use, or are you open to suggestions? (e.g., Amazon reviews, MovieLens, proprietary)
2. Do you have a preference for Qwen3 model sizes (e.g., 0.5B, 1.8B, 4B, etc.) or should I suggest the largest feasible one for a 24 GB 4090?
3. Are you already using any specific libraries like Hugging Face Transformers or PEFT, or are you open to the best current options?
4. Would you prefer your final setup to be in Python scripts, Jupyter notebooks, or any other format?

Let me know and I’ll kick off the deep research right away!

---

---

Great, I’ll get started on deep research to design a practical LoRA fine-tuning setup for a Qwen3 model (≥4B) on the MovieLens dataset (and optionally the Delivery Hero dataset) for binary recommendation. I’ll focus on real GitHub repos and blog posts that use Qwen or similar models with PEFT/QLoRA on binary classification or recommendation tasks.

I’ll also provide concrete data formats, LoRA config, training loop plans, engineering tips, monitoring strategy, and an end-to-end recipe that will work on a single RTX 4090.

You can keep chatting with me while I work on this—I’ll let you know once it’s ready!

## LoRA Fine-Tuning Qwen3 for Recommendations: End-to-End Blueprint

## 1\. Open-Source LoRA Examples for Qwen3 and Recommendation Tasks

To ground our blueprint, we surveyed open-source repositories and examples of LoRA fine-tuning on Qwen models and on recommendation-style tasks. Below is a curated list of relevant projects, with model details, libraries, and dataset/task information:

### Qwen-Native LoRA Fine-Tuning Examples

- **Qwen 3 Fine-Tuning (Official)** – *Model:* Qwen3 (available in 4B, 7B/8B, 14B dense, plus MoE variants up to 30B+). *Libraries:* Hugging Face Transformers + PEFT, often with DeepSpeed or Accelerate. *Task:* General **instruction/chat SFT** (supervised fine-tuning) on custom datasets. *Format:* Data in Qwen’s **ChatML** JSONL format: each sample contains a list of `{"role": ..., "content": ...}` messages (system/user/assistant). *Hardware:* The official example supports single or multi-GPU; QLoRA (4-bit quantization) is recommended for larger Qwen models. A single 24 GB RTX 4090 can handle **Qwen-7B or Qwen-14B** with LoRA by using 8-bit/4-bit base precision and techniques like ZeRO or gradient checkpointing. (This is the blueprint we will adapt for our task.)
- **Hugging Face Optimum (Qwen3-8B LoRA)** – *Model:* Qwen3-8B (dense model). *Libraries:***Optimum-Neuron** (for AWS Trainium) + Transformers + PEFT. *Task:***Recipe generation** (instruction fine-tuning on a cooking recipes dataset). *Format:* Chat-style prompts: user asks “How can I make \[RecipeName\]?”, assistant outputs the recipe steps. The training script uses Qwen’s chat template and adds an end-of-sequence token after each answer. *Hardware:* Targeted at AWS Trainium (16 accelerators) but the configuration (LoRA `r=64`, `alpha=128`, dropout 0.05) can be scaled down for a single GPU by lowering `r`. The example used FlashAttention2 for 4k+ context and showed that even Qwen3-8B can be fine-tuned with LoRA on one GPU if batch size is kept low (`per_device_batch_size=1` with grad accumulation).
- **Ivan’s Qwen3 LoRA Alignment Repo** (by *ivan-digital*) – *Model:* Qwen3-14B (dense) with long context (8k–16k tokens). *Libraries:* HF Transformers + PEFT (with custom training loop for KL-regularized SFT and DPO). *Task:***Instruction following** (with alignment techniques). *Format:* JSONL with chat role messages for instructions and answers. *Key features:* Uses **LoRA on Qwen3** (e.g. rank 16, α=16) applied to attention projection layers to **minimize forgetting** of base capabilities. They report that modest LoRA (r=16) and a KL penalty keep the model’s original knowledge intact while learning new formats. *Hardware:* The provided scripts are optimized for a **single GPU**; they enable gradient checkpointing and report stable training of Qwen3-14B with LoRA rank 16 on 24 GB memory. This shows that fine-tuning a Qwen3 14B on one 4090 is feasible with 4-bit precision and memory optimizations.
- **Unsloth Qwen3 Fine-Tuning** – *Model:* Qwen3 family (8B, 14B dense; 30B MoE “A3B” etc.). *Libraries:***Unsloth** (an open-source fine-tuning toolkit) with FlashAttention 2 and custom optimizations. *Task:* Any SFT or instruction tuning (demonstrated on Alpaca and reasoning datasets). *Format:* Accepts the same role-based chat format or instruct format. *Key features:* Unsloth’s pipeline is highly optimized for Qwen3 – it allows **quantized fine-tuning with minimal accuracy loss** and up to 8× longer context windows. For example, Qwen3-8B fine-tuning runs ~2× faster and uses >70% less VRAM with Unsloth (enabling Qwen3-30B to fit in ~17.5 GB). Unsloth inserts LoRA adapters in all linear layers (attention *and* FFN) by default. *Hardware:* A single 24 GB GPU can fine-tune Qwen3-14B or even a 30B MoE model under Unsloth’s optimizations. This is a good reference if you need to push model size: it demonstrates Qwen3-30B (MoE) LoRA with batch 2 and grad\_acc 4 (global batch 8) on 24 GB.

### Non-Qwen Examples (Adaptable to Qwen3 Recommendation)

- **LLM-RecSys (Hemang)** – *Model:* Mistral-7B and “TinyLlama” (a smaller LLaMA variant, ~1–3B). *Libraries:* Hugging Face Transformers + PEFT (using **QLoRA** via bitsandbytes) and **Unsloth** for training scripts. *Task:***Personalized product recommendation** – predict a user’s next purchase. They fine-tuned on a subset of the **Amazon Reviews 2018** dataset (user–product interactions). *Format:* They designed a **novel prompting strategy**: each training sample is a three-part instruction for the LLM – an **Instruction** (what to do), an **Input** (the user’s interaction history in natural language form), and the **Output** (the target product or a “yes/no” indicator). For example, a prompt might say: *“Recommend whether the user would buy the next item. History: \[item1, item2,...\]. Candidate: \[itemX\].”* and the output is “Yes” or “No.” This structure gave the model context and a clear expected answer format. *Results:* Using LoRA/QLoRA, they reportedly achieved **\>98% accuracy** in predicting next purchases with the fine-tuned TinyLlama – highlighting that a specialized smaller model can outperform a larger one if well-fine-tuned for a specific task. *Hardware:* Training was done on a single GPU with 16–24 GB (the project emphasizes efficiency). Mistral-7B in 4-bit with LoRA (r=16) fits in 16 GB with batch size 8, so on a 24 GB 4090 one could use a larger model or batch. This repo is highly **adaptable to Qwen3** – you can apply the same prompting and LoRA setup to Qwen3 models.
- **Multilabel Classification with Mistral-7B** (Hugging Face Blog by *Lukas*) – *Model:* Mistral-7B (an LLaMA2-7B derivative). *Libraries:* HF Transformers, **bitsandbytes (4-bit quant)**, **PEFT**. *Task:***Text classification** (multi-label classification of research paper topics). *Format:* Each input is a text (title & abstract) and labels are multiple yes/no tags. They fine-tuned by converting Mistral to an `AutoModelForSequenceClassification` with LoRA. *Relevance:* This example shows how to use LoRA for **binary/multiclass classification** rather than generative chat. They applied LoRA to Mistral’s attention layers (`q_proj, k_proj, v_proj, o_proj`) with `r=16, α=8, dropout=0.05, bias='none'`, and froze the feed-forward layers (thus saving memory by not LoRA-tuning them). The classification head was a small fully-connected layer automatically added to the model (excluded from quantization and trained alongside LoRA). *Hardware:* Fine-tuning 7B in 4-bit on a single GPU (16 GB) was successful with batch size 8. On a 24 GB 4090, this approach can be extended to Qwen3-7B or 14B. This example is “ **non-Qwen but adaptable** ”: to use with Qwen, one could similarly load `AutoModelForSequenceClassification` with a Qwen base and apply LoRA via PEFT – useful if treating recommendation as a classification task with limited outputs.
- **Adapter4Rec (WSDM’24)** – *Model:* Variety of backbones (possibly BERT, Visual encoders, and could be extended to LLMs). *Libraries:* PyTorch + `loralib` (the LoRA library). *Task:***Multi-domain recommendation** from raw item modality (text/images) using adapter tuning. *Insight:* Though not LLM-specific, this research shows that **adapter-based tuning** (like LoRA) can achieve competitive performance in recommender systems across domains. They explore *where* and *how* to insert adapters in a model for recommendation tasks and confirm that adapters can match full fine-tuning performance in TransRec settings. It provides guidance on using LoRA in non-LLM recommenders and could inspire where to apply LoRA in an LLM (e.g., in language or multimodal layers). *Hardware:* Their experiments were on standard GPUs; applying it to Qwen3 on a 4090 is feasible, as the LoRA technique is lightweight. (This is more research-focused, but underscores that **LoRA is a proven strategy for recommender systems** as well.)

**Summary:** We have solid references that Qwen3 models can be LoRA-fine-tuned on one GPU, and that LoRA has been applied successfully to yes/no recommendation tasks. In particular, the **Qwen official SFT script** and **Hemang’s LLM-RecSys project** will be our primary guides: they show how to format data (chat or instruction style) and how to configure LoRA for a binary recommendation goal.

## 2\. Designing the Data Format for a Qwen3 Recommendation Model

We need to feed the model each user’s **purchase history + a candidate product** and train it to output a binary label (“yes” or “no”). There are two general strategies for formatting this: (A) an **instruction/chat prompt style**, and (B) a more direct **classification style** where the label is the final token. We’ll design both and discuss their pros/cons.

### Format A: Prompt/Chat Style (Instruction-Following)

In this format, we present the task as a natural language question to the model (as if instructing a chatbot or assistant). This can be done in pure instruction form or using Qwen’s chat roles. The model will generate “Yes” or “No” as a **natural language answer**. For example:

- **Instruction-style JSONL (Alpaca/Dolly format):**
	Each sample has an `"instruction"` (telling the model what to do), an `"input"` that includes the **history and candidate** in textual form, and the desired `"output"`. We clearly separate the user’s data from the instruction. The model sees a prompt like: *“Purchase history: \[A, B, C\]. Candidate: \[X\]. Should we recommend?*” and must answer “Yes” or “No.”
- **Qwen ChatML format:** Since Qwen-3 is chat-oriented, we can alternatively use role tags:
	This follows Qwen’s data schema. We set a brief system prompt to clarify the task (and to constrain the answer style), then the user prompt lists the history and candidate, ending with a direct question. The assistant’s content is `"Yes"` or `"No"`.

**Pros:** This approach leverages the LLM’s natural instruction-following ability. It’s **interpretable** (we see the reasoning in plain text if the model were to explain itself, though here we force a concise answer). It aligns with how chat models are typically fine-tuned, which may help Qwen3 learn the task efficiently. Another benefit is we can embed additional context easily: e.g., including **product titles, categories, or descriptions** instead of opaque IDs. Using descriptive text (like product names) allows the model to use its world knowledge and semantics (e.g., knowing that people who bought a PS5 likely want PS5 games). If only IDs are available, a prompt style format can still work (the model will treat IDs as tokens and learn correlations), but using titles makes the task more natural to the language model.

**Cons:** There’s a risk of the model generating more than just “Yes”/“No” – since in a freeform response it might elaborate (“Yes, because they have similar items…”). We can mitigate this by instructions (“answer with a single word”) or by how we structure training (see below on label tokens). Another consideration is **label leakage**: because the model is generating the answer, we must ensure the prompt doesn’t implicitly give away the answer. In our format, it doesn’t – the model must infer from history. Also, the token “Yes” or “No” should not appear in the prompt, only in the answer segment, so the model truly has to predict it.

**Sequence length & memory:** Prompt style can be verbose. For a long purchase history, the input prompt might become large. On a single 4090, we should keep sequence lengths reasonable (to fit in memory and to ensure faster training). Qwen3 supports very long contexts (up to 8k or more with FlashAttention), but that’s not necessary for most recommendation histories. It’s often better to truncate or summarize the history beyond a certain length:

- **Truncation:** We can include only the **N most recent items** in the history. Recency often matters most in recommendations, and it keeps input short. For example, use the last 20 purchases or last 1–2 years of activity. This keeps sequences say under ~200 tokens (if each product title is a few tokens).
- **Sampling/selection:** If a user has a very rich history (hundreds of items), another strategy is to **sample a subset** of representative items (ensuring we cover diverse interests) or **filter** by relevance to the candidate (if we know categories). However, this is complex; a simpler rule like *“take the last N items”* usually works well.
- You could also use an **aggregate description** (“User is an avid gamer and tech enthusiast”) but that would require generating such summaries, which is beyond our scope (and could introduce errors).

**Example truncation:** If a user bought 100 items, we might only take the 20 most recent. We should also impose a max token length for the combined prompt. For instance, set `model_max_length = 512` or `1024` tokens for training – this ensures even if some histories are long, they’ll be cut off. The Qwen data loader or our custom collate can slice the history if it’s too long (possibly with a marker like “… and more” to indicate truncation).

**Label handling:** To ensure stable training, we want the model’s output to be exactly “Yes” or “No”. We will include the correct label in the **assistant’s response** during training. It’s important to append an end-of-sequence token after the label in training data. For example, the assistant content could be `"Yes<\/s>"` (where `<\/s>` is the EOS token). By doing this, the model learns to stop generation after producing the label. During inference, we can prompt the model and either:

- Use greedy decoding with a max new tokens = 1 (so it generates one token, which will hopefully be “Yes” or “No”), **or**
- Constrain the decoding vocabulary to just the tokens for "Yes" or "No" (some generation APIs allow specifying allowed tokens). In practice, we can simply stop after the first token for this task.

**Prompt Format Variant:** Instead of asking as a question, one could format it as a statement for classification. For example: *“History:... Candidate:... Prediction: **Yes**.”* and train the model to fill in “Yes” or “No” after “Prediction:”. This is akin to a cloze prompt and ensures the model outputs only the word. Both approaches are viable. The key is consistency: choose one prompt template and use it for all training examples so the model learns the pattern.

### Format B: Classification Style (Direct Label Prediction)

In this format, we frame it as a classical classification problem for the model’s final token(s). We will use the language model in a way that the **last token corresponds to the label**. This could be done by leveraging the model’s generation or by using a classification head.

**Method B1: Generative with Label Token** – We can structure the prompt such that the very next token to generate is the label. For example:

- *Prompt as text:*`"User purchased: [list]. Candidate: [X]. Recommend? ->"` and we train the model that the next token after the arrow is either “Yes” or “No”. Essentially, the model treats it like a fill-in-the-blank. Another example: `"History: ... Candidate: ... Answer (Yes/No):"` and the model must output either "Yes" or "No" as the completion.

In implementation, this means for each training sample we concatenate the prompt and the label, but we set up the labels tensor so that the prompt part is **ignored in loss** and only the final token (or final word) is supervised. With the Hugging Face Trainer, this can be done by providing `labels` that are `-100` for all input prompt token positions and the actual label ID for the answer token position. The model then effectively learns to predict that single token label given the preceding context.

**Method B2: Sequence Classification Head** – Alternatively, one can use `AutoModelForSequenceClassification` for Qwen. In this setup, the model encodes the whole sequence (history + candidate text) and then a small **classification head** predicts yes vs no. This is what the Mistral-7B example did for multi-label classification. We would format each input as, e.g., `"History: item1; item2; ... Candidate: itemX"` (without any explicit question) and let the classifier output logits for the two classes. This doesn’t use the LM’s generation for output at all – it treats it like any BERT-style classifier. During fine-tuning, only the LoRA adapters and this classifier layer are trainable (base model frozen except LoRA). The output labels would be 0/1 or indices for “no”/“yes”.

**Pros of Classification style:** It **forces a restricted output**. There’s no risk of the model writing a long response or deviating, since either we stop after one token or use a classifier head. This can make training more stable for a binary task. Also, evaluation is straightforward: the model gives a probability for yes vs no which we can directly use for metrics like AUC. With the classification-head approach, you inherently get a probability distribution over labels from the logits, which is convenient for threshold tuning.

**Cons:** If using the classification head (Method B2), we lose some of the “chat” capability – the model isn’t generating in natural language, so this fine-tuned model is purely a classifier. (However, since we plan to use LoRA, the base model’s weights stay intact; we could disable the classifier layer and still use the LoRA-equipped model for generation if needed, but the LoRA might have learned only to output those tokens.) If using the generative single-token method (B1), it’s quite similar to the prompt style in implementation complexity (we have to manage loss masking), and it still requires careful prompt design to ensure the answer is a single token. Also, “Yes” and “No” in many tokenizers are single tokens (e.g., in English, likely they are). We should verify that – if not, we might define special tokens or just accept a multi-token output like “No</s>” (where the second token is EOS). Usually, they are single tokens in GPT-style vocabularies.

**Sequence length:** The classification format is *efficient* with sequence length – we include just the history and candidate, plus maybe a short prompt string, and no extra role or instruction verbiage. So it tends to be shorter than the chat style. We similarly need to truncate long histories here. For example, if the history list is too long to fit in 512 tokens, we cut it. We could also drop very old interactions. Because this format is not “explaining” the task in natural language (we assume the model knows from fine-tuning that the text is structured as history → candidate → label), we must rely on the fine-tuning data to teach the model how to interpret the input. That’s fine given enough training examples, but it’s less flexible than having an instruction.

**Example JSONL record (classification style):**

If using generative B1:

This single string would be tokenized. We ensure the '?' and 'Answer:' part are in the prompt so the model expects a "Yes" or "No" next. During training, we'd mask all tokens except "Yes" for loss.

If using a classifier head (B2), our dataset might be something like:

(where `1` = yes, `0` = no, and we use a separator like "||" or some token to delimit history from candidate). We would then feed these to the model as input\_ids with labels for the classification head.

**Pros/Cons Summary:** Classification formats are efficient and precise for outputs. The downside is they treat the task in a more “narrow” way, potentially not leveraging the full reasoning ability of the LLM. In contrast, prompt style might allow the model to *internally* reason (especially if we allowed it to output reasoning – but here we don’t want that visible). However, since our final goal is just yes/no, a concise format is usually sufficient.

**Recommendation:** We can actually **combine the benefits**: use a prompt style input (with clear instruction) but constrain the output to yes/no by design. The **Format A** examples above do exactly that – the question explicitly asks for “Yes or No” and the assistant answer is one word. This way we maintain an instruct prompt (which the Qwen model is used to) yet get a simple output space.

To keep sequences short and training efficient on a 4090:

- Limit histories to a fixed token length or item count (e.g., truncate to last 50 items or last 256 tokens of history).
- Use titles or brief descriptions for items rather than long texts. If product titles are long, consider using shorter identifiers *plus* category (e.g., “Sony PlayStation 5 (Video Game Console)” – the category in parentheses might help the model generalize, then we might not need every word of a title).
- If using JSONL with roles, the overhead of `{"role":..., "content":...}` adds tokens during parsing but not in the model input (the tokenizer will see only the actual content strings and special tokens for roles if any). Qwen’s tokenizer uses `<|system|>`, `<|user|>`, `<|assistant|>` tokens for roles in ChatML format. This adds a few tokens per message but it’s negligible compared to a long history list.

In summary, **two concrete formats** to consider for our case:

1. **Chat Prompt format:** (Recommended for Qwen-Chat model) – Use system message to set context (“You are a recommender…”), user message contains history and candidate with a direct question, assistant message is the yes/no label. Example provided above. This is very explicit and leverages Qwen-Chat’s training format.
2. **Instruction + Input format:** (Works for base Qwen or chat) – Combine the fixed instruction and history as shown in the JSON example with `"instruction"` and `"input"`. This is essentially how you’d fine-tune an instruct model. It’s easy to generate such data and you can include an “Answer:” cue to indicate where the model should output the label.

Both formats require careful **consistency** in training data. We should **not mix** different prompt styles in one training run, as that might confuse the model. Pick one format and apply it to all samples. Given Qwen3’s chat heritage, the ChatML format might integrate most smoothly if we use a Qwen-Chat model. If we use the base Qwen (non-chat), the simple text instruction format (with e.g. “Answer: Yes”) could be better since the base model might not have the special role tokens.

Finally, document the chosen format in the training config (so it’s clear how to prompt the model later), and possibly include a few **prompt examples in the model card** after fine-tuning, so you remember how to query it for recommendations.

## 3\. Concrete LoRA Training Plan for Qwen3 on a Single RTX 4090

Now we outline a step-by-step plan to fine-tune a Qwen3 model using LoRA on one 24 GB GPU. We assume a dataset on the order of 50k to a few million (which is realistic for recommendation data). The plan covers model selection, precision (to fit in memory), LoRA setup, hyperparameters, and training implementation details.

### Model Choice & Precision

**Choose a Qwen3 model size** that balances capability and resource limits. Given the user’s note (“at least 4B for better result”), we should go with **Qwen3-7B or larger**. Qwen3 has variants around 4B, 8B, 14B, etc. A **7–8B model** is a great starting point – it is small enough to fine-tune quickly, but large enough to capture complex patterns. If maximum recommendation quality is needed and the data is rich, consider the **Qwen3-14B** model (which will likely yield higher accuracy/coverage due to more parameters). On a single RTX 4090:

- **7B dense model** can be fine-tuned in 16-bit, but we will use lower precision to leave room for batch size.
- **14B dense** model in 16-bit would be ~28 GB (too high), but with **4-bit quantization (QLoRA)** it can fit comfortably. Many have fine-tuned 13B LLMs on 24 GB GPUs using QLoRA, so Qwen3-14B (similar size) is feasible.

**Precision setup:** We will use the popular **BitsAndBytes** library to load the base model in 4-bit or 8-bit precision. A common approach is:

This loads Qwen3 in 4-bit precision, using bfloat16 for math (for stability) – a typical QLoRA setup. We then apply LoRA on top of this (LoRA will be in FP16/BF16). The **memory estimate**: Qwen3-7B in 4-bit uses ~4–5 GB GPU RAM for weights, Qwen3-14B ~8–10 GB. The optimizer states for LoRA are very small (negligible compared to model). So with 24 GB total, we have plenty of headroom for activations and a decent batch size.

If using **bf16 vs fp16**: We prefer **bf16** (bfloat16) if your GPU supports it (the 4090 does) because it has a larger exponent range, reducing overflow risk. Hugging Face Accelerate/Trainer can be told to use bf16 (`--bf16` flag or `training_args.bf16=True`). This avoids certain instabilities in training large models. If bf16 isn’t available, fp16 is fine but maybe enable grad clipping or scale loss if needed.

For **Qwen3 MoE** models (like 30B A3B), quantization is even more crucial. Unsloth demonstrates 30B-MoE in 4-bit fits in 17.5 GB. But for simplicity, we’ll assume a dense model variant.

**Model variant:** Use the **Qwen-Chat** model if you plan to use chat format prompts. For example, `Qwen/Qwen3-7B-Chat` if available. This comes with conversational tuning which might help it understand the prompt style out-of-the-box. However, since we are further instruction-tuning it on a specific task, using the **base model** `Qwen3-7B-Base` is also reasonable – especially if we do a classification style format (no need for it to follow human-like instructions beyond our prompts). Both will work with LoRA. If uncertain, I’d lean towards the chat model and use a system message at runtime to enforce concise answers.

### LoRA Configuration

We’ll use the **PEFT library** to configure and apply LoRA. Key settings in `LoraConfig` include:

- `target_modules`: Which parts of the model to insert LoRA weights into.
- `r` (rank): The bottleneck dimensionality of the LoRA update matrices.
- `lora_alpha`: The LoRA scaling factor (often set proportional to `r`).
- `lora_dropout`: Dropout on the LoRA updates.
- `bias`: Whether to train any bias terms.

**Target modules for Qwen3:** In Transformer-based LMs, LoRA is commonly applied to the **attention projections** (queries, keys, values, and sometimes output projection) because these are large weight matrices and influential for model behavior. Many works (including the QLoRA paper) found LoRA on **W<sub>q</sub> and W<sub>v</sub>** (query and value) gives a good trade-off of quality vs. parameters. However, recent examples with bigger models (and Unsloth’s default) apply LoRA to **all projection layers in both self-attention and feed-forward** blocks. For Qwen3:

- The attention layer typically has weight matrices for Q, K, V, and the output (O or proj).
- The feed-forward (MLP) layer has an “up” projection (expand dimensions), a nonlinear activation, then a “down” projection (project back to model dim). In some architectures there’s also a “gate” projection (if using gated activation like GLU).
- Qwen’s architecture is similar to LLaMA’s, so we expect names like `"q_proj", "k_proj", "v_proj", "o_proj"` for attention and `"gate_proj", "up_proj", "down_proj"` for the MLP (these were confirmed in the Reddit/Unsloth references).

Thus, a safe `target_modules` list is:

This matches what community fine-tuners used for Qwen3. If memory is a concern, you could omit some (e.g., only do Q, V, and perhaps Up/Down). But given LoRA’s param count is small, we can include all to give the adapter more expressive power. (The Optimum-Neuron example even included the embedding matrix in LoRA, but that’s uncommon; we can skip embedding unless we need domain adaptation for new tokens.)

**Rank (`r`):** This is a crucial hyperparameter. A higher rank means the adapter can capture more complex transformations, but also uses more memory and **risks overfitting** or overriding the base model’s knowledge. Prior findings with Qwen3 show:

- *Small ranks (e.g. 8)* often suffice and preserve the model’s original capabilities.
- Increasing to rank 32 or 64 made the Qwen3 model forget some of its prompt-handling tricks and even produce worse outputs in that anecdotal test.
- Rank 8 was the “sweet spot” for a behavior tuning task on Qwen3-8B – it directed the model without breaking it.

For a binary recommendation task, we are not heavily altering the model’s language style or general knowledge – we’re just teaching it a new prediction behavior. This typically does **not** require a very high rank. **Start with `r = 8`** as a baseline. It will be very memory-efficient (for Qwen3-7B, rank-8 across the above modules is on the order of only ~5 million trainable params or less, which is nothing for a 7B model). If you find the model isn’t learning the task well (e.g., if training accuracy/recall is low, indicating underfitting), you can consider raising to `r = 16`. It’s unlikely you’d need beyond 16 or 32 for this task. And as the Reddit experiment showed, too high can degrade performance.

**Alpha (`lora_alpha`):** This scales the LoRA updates. A common default is to set `alpha = r` (or sometimes `alpha = 2*r`). In one example, rank 8 used alpha 16 (2x the rank), and rank 16 often uses alpha 16 or 32. You can think of alpha as a way to boost the initial output of LoRA layers; however, after training, it can be somewhat absorbed into the learned weights. We’ll set `alpha = 16` for r=8 (or generally `alpha = 2*r`). This matches community settings that worked for Qwen3. If using r=16, alpha 32 is reasonable. (PEFT’s default alpha is 32, which is fine for r in \[8,32\]).

**Dropout:** Adding dropout to LoRA layers can act as regularization. The default 0.1 is often a bit high; many use **`lora_dropout = 0.05`** or even 0.0 if the dataset is large and diverse. We recommend 0.05 initially – it helped in both the Mistral and Qwen3 fine-tunes to prevent overfitting. If you see the model underfitting (loss not decreasing much), you could remove LoRA dropout in a later run.

**Bias:** LoRA can also allow training of biases in the model (options: 'none', 'all', or 'lora\_only'). Most reports set `bias='none'` (do not train any biases). This keeps the number of trainable parameters minimal and avoids some potential drift in output distribution. For a classification task, one might argue for adjusting final layer norm biases or so, but it’s usually unnecessary. We’ll keep biases frozen (`bias='none'`).

**LoRA config example:**

For sequence classification (if you go that route), use `task_type='SEQ_CLS'` which ensures LoRA layers are applied to the underlying base model, not the classifier layer (which we might train normally). In causal LM, `'CAUSAL_LM'` is appropriate.

**Trade-offs (`r` memory vs capacity):** On a 4090, the memory impact of different `r` is relatively small:

- Each LoRA matrix has shape (hidden\_dim, r) or (r, hidden\_dim). For Qwen3-7B, hidden\_dim ~ 4096. So r=8 yields two small matrices of size 4096×8 and 8×4096 per target module (plus maybe minor overhead). That’s about 65k parameters per matrix, maybe ~130k per module pair. With 7 modules targeted, ~910k parameters (times 2 for two matrices per module) ~1.8M \* 2 (since each weight has grad and optimizer states in FP16) ~7.2M floats – trivial for GPU memory. r=16 quadruples this to ~29M floats; still under 0.06 GB.
- So **memory is not a big issue**; the bigger trade-off is *capacity vs preservation*. A smaller r will force the model to only make minimal, low-rank adjustments – which is often enough for a binary task and helps keep the model’s original text generation style intact. A larger r can capture more nuanced patterns (if the user features and item features are very complex, a higher rank might fit them better), but it might also start to mimic full fine-tuning and override behavior.
- Considering we have potentially millions of training examples, the model can likely learn the needed patterns even with low rank. I would start with r=8 or 16. **Monitor validation performance** – if it saturates quickly or you notice the model output is overly simplistic, r might be too low; you could try r=16 next. Conversely, if the model starts outputting “Yes” to everything (overfitting), reducing rank or adding dropout could help.

### Training Hyperparameters (4090-specific tuning)

**Batch size and Gradient Accumulation:** With a 24 GB GPU, we want to maximize throughput without OOM. The sequence length will determine how many samples we can fit per batch. Assuming a max sequence of 256 or 512 tokens:

- For Qwen-7B in 4-bit, one forward pass with seq 512 and batch size 8 might fit (the Mistral example did batch 8 on 16GB at seq ~200). We can likely do **batch size 8–16 per step** on 24GB for 7B. For 14B, batch size 2–4 might be the limit in 4-bit.
- We can use **gradient accumulation** to simulate a larger batch if needed. For example, if we set `per_device_train_batch_size = 4` and `gradient_accumulation_steps = 4`, the effective batch is 16.
- Aiming for an effective batch of **16 to 64 samples** is reasonable for classification tasks. Larger batches can stabilize training for yes/no problems (which can be noisy if batch is too small). If the dataset is huge (millions), large batch might also speed up epoch time. But be mindful: extremely large batch combined with high learning rate can sometimes converge to a sharp minimum too quickly or get stuck. So don’t go beyond what the GPU can handle by more than ~4–8× via accumulation.

Given the 4090’s power, I’d try **batch 8** (no accumulation) for 7B. For 14B, perhaps **batch 2 with grad\_acc 4** (to effectively have 8) – this is exactly what one Qwen3 LoRA user did.

**Learning rate schedule:** LoRA adapters often tolerate a higher learning rate than full-model fine-tuning, because we’re training far fewer parameters (and not destabilizing all weights). Common ranges:

- For LoRA on LLMs, LR around **1e-4 to 2e-4** is often used. The QLoRA paper used ~2e-4 for 7B models fine-tuning on instruction data, and around 1e-4 for 13B.
- If using 4-bit (QLoRA), sometimes slightly lower LR like 5e-5 to 1e-4 is suggested, but practitioners have reported success with 2e-4 even in 4-bit. It likely depends on the scale of the data.
- We can start with **LR = 1e-4** (0.0001). It’s a conservative mid-range choice. If training loss plateaus too high, we can bump to 2e-4. If we notice val loss diverging or the model overfitting quickly, we might lower to 5e-5.

Use a **warmup** at the start: e.g., 1–2% of total steps of linear warmup. If we train for, say, 10k steps, first 100–200 steps warmup from 0 to 1e-4 then continue. This avoids shocking the model with a high LR immediately.

As for **scheduler**: either a **cosine decay** or **linear decay** to 0 by end of training. Cosine is popular for fine-tuning as it anneals gently. Alternatively, a linear decay with maybe 10% of steps for warmup can work. We can specify `--lr_scheduler_type cosine --warmup_ratio 0.02` in HF Trainer, for example.

**Epochs or steps:** This depends on dataset size:

- If we have a few million examples, one epoch is huge and likely enough to converge. In recommendation, often multiple epochs can lead to overfitting because the model will start memorizing patterns (especially if many negative samples are repeated or easy to guess).
- I’d suggest using **early stopping on validation** rather than a fixed large number of epochs. Perhaps start with **1 epoch** through the data and see metrics. If the dataset is smaller (50k), you might do 2–3 epochs.
- For millions of samples, even 1 epoch could be 1e6/ (batch\_size\*grad\_acc) steps, which might be tens of thousands of steps. The model will likely saturate accuracy before finishing one epoch if the signal is clear.
- As a guideline: you could set a **max\_steps** budget, like 20k steps, and evaluate every epoch or every N steps. Use the validation metrics to decide if more training is needed. It’s important not to severely over-train since a binary classifier can start outputting a single class for everything if pushed too far (you’d see training loss still decreasing but val metrics getting worse).

**Weight decay:** Usually, LoRA weights are small and regularization through weight decay isn’t critical. But it doesn’t hurt to use a little. The Mistral classification used `weight_decay = 0.01`. We can similarly set **0.01**. Note that only the LoRA parameters and any classifier head parameters (if present) will be subject to decay; the base model’s weights are frozen (so their decay doesn’t apply).

**Gradient clipping:** It’s wise to clip gradients at some max norm (like 1.0) to prevent spikes, especially with fp16 training. Hugging Face Trainer can do this via `max_grad_norm`. We can set `max_grad_norm = 0.3` to `1.0`. LoRA training usually doesn’t explode, but with a high LR it could.

**Mixed precision settings:** As mentioned, use `bf16` if possible for speed and stability. Set `fp16=True` only if bf16 not available. The Trainer will handle casting. With 4-bit base model, ensure `prepare_model_for_kbit_training(model)` is called – it tweaks some layers (like embedding gradients, layer norms) to be fp32 to avoid certain issues.

**Max sequence length:** We touched on this in data format – decide on a length (maybe 256 tokens) that covers most cases. If using chat format with role tokens, perhaps 256–512 to accommodate up to, say, ~20 history items + prompt. We will set `model.config.max_position_embeddings = 512` (if not already) or pass `--model_max_length 512` to the Trainer. If histories are generally short, we could lower to 256 for slightly faster training. Keep in mind Qwen3 can handle long contexts, but training on unnecessarily long sequences (with mostly padding) is wasteful.

To summarize the hyperparams:

- **Learning Rate:** 1e-4 (with warmup ~2% and cosine decay). Possibly adjust after initial runs.
- **Batch Size:** ~8 (or as high as fits, up to 16) per GPU, with grad accum to reach ~32 if needed.
- **Epochs:** 1–3 (monitoring val metrics for early stop).
- **Weight Decay:** 0.01.
- **Max Seq Length:** 256 (truncate history accordingly).
- **Precision:** bf16 for compute, 4-bit for model weights.
- **Grad Checkpointing:** Enable if needed to fit memory, especially if you attempt longer sequences or a 14B model. This trades speed for memory. Qwen3’s HF implementation likely supports it (`model.gradient_checkpointing_enable()`). In prior Qwen LoRA training, enabling checkpointing was key to fitting larger models on a single GPU.

### Implementation Details: Training Loop / Script Outline

We have two main implementation paths:

1. Use the **Hugging Face Trainer** with PEFT – easier, high-level.
2. Use a custom training loop (possibly leveraging libraries like TRL or Unsloth for convenience).

**Using Transformers Trainer + PEFT:** This is straightforward and likely sufficient.

- Prepare the dataset as a `datasets.Dataset` or PyTorch `DataLoader`. If using JSONL, you can load it with `datasets.load_dataset("json", data_files=...)`.
- Write a preprocessing function to tokenize each prompt-answer pair. Important: if doing causal LM, you’ll concatenate prompt and answer and create proper `labels`. For example, for each entry, you can do:
	If using the ChatML format with roles, you might use Qwen’s `tokenizer.apply_chat_template` as in Optimum’s example, which handles inserting special tokens. Then just ensure the last token is the correct label + EOS.
- Define the LoRA model as shown earlier and the training `TrainingArguments`.
- **Data Collator:** If using padding, a simple data collator with `tokenizer.pad` will suffice (since we want dynamic padding). For causal LM, use `DataCollatorForLanguageModeling` with `mlm=False` if you didn’t manually set labels as above. If we manually prepare labels, a custom collator that just converts lists to tensors and pads is fine.
- **Metrics:** Implement a `compute_metrics` that computes accuracy, F1, etc. For binary:
	```python
	pythondef compute_metrics(eval_pred):
	    predictions, labels = eval_pred
	    # If using sequence classification, predictions are logits [batch,2].
	    # If using generative with soft labels, we might get token ids.
	    if predictions.shape[-1] == 2:  # sequence classification scenario
	        probs = softmax(predictions, axis=-1)
	        preds = np.argmax(predictions, axis=-1)
	    else:
	        # If generative, decode or take the token directly.
	        preds = predictions  # (depending on how we gather eval outputs)
	    labels = labels.flatten()
	    accuracy = (preds == labels).mean()
	    # F1 for positive class perhaps:
	    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
	    auc = roc_auc_score(labels, probs[:,1] if 'probs' in locals() else preds)  # for generative, we'd need probability of "yes".
	    return {"accuracy": accuracy, "f1": f1, "precision": prec, "recall": rec, "auc": auc}
	```
	We will log metrics such as accuracy and AUC which are critical in recommendation.
- Initialize Trainer:
	This is a high-level approach. The trainer will handle accumulation, mixed precision, etc. We use the `adamw_bnb_8bit` optimizer from bitsandbytes for memory efficiency (it will keep optimizer states in 8-bit). We log to Weights & Biases for monitoring (we’ll set that up in section 5).

**Using Unsloth or TRL:** If you prefer, Unsloth provides a simplified fine-tuning interface for Qwen. For example, Unsloth’s SFTTrainer can wrap a lot of these steps. In Unsloth, you might do:

Under the hood, Unsloth will apply flash attention, gradient checkpointing, etc. automatically. If you’re new to it, however, the HF Trainer approach is more transparent. TRL (Transformer Reinforcement Learning library) is more relevant for RLHF, but it also has a trainer for SFT that could be used similarly to HF’s Trainer.

**Memory/OOM mitigation:** On a 4090, if you face OOM issues:

- **Gradient Checkpointing:** This should be your first tool. It saves memory by not storing all intermediate activations. Enable via `model.gradient_checkpointing_enable()` before training or `TrainingArguments.gradient_checkpointing=True`. It can reduce memory usage by ~30-40% at the cost of some speed.
- **Reduce Batch Size:** Lower `per_device_train_batch_size` first (e.g., from 4 to 2) before truncating sequence length, because preserving more of the sequence (history information) might be important for model accuracy. Increase grad accumulation if needed to maintain an effective batch.
- **Shorten sequence length:** If absolutely needed, cut `max_length`. For instance from 512 to 256. This will linearly save memory and speed. The trade-off is losing some older history info, but if you have to do this, it likely means that very long histories aren’t manageable anyway.
- **Use 8-bit instead of 4-bit if issues:** Actually 4-bit saves more memory than 8-bit, but 4-bit quantization can sometimes cause slight slowdowns due to how it packs data, and in rare cases instability. If 4-bit is problematic, 8-bit is still a big memory saver over full precision and might be more stable. However, in practice 4-bit QLoRA for Qwen is known to work.
- Ensure **flash attention** is on: Qwen’s architecture might support FlashAttention (some libraries require you to explicitly enable it or install a flash-attn package). FlashAttention2 especially helps with long contexts. In our plan, we probably don’t need extremely long context, but if using >1k tokens, it’s worth enabling to reduce memory usage of self-attention by kernel optimization.
- **Monitor GPU memory** (we’ll discuss in section 5) – sometimes fragmentation can cause OOM even if usage looks okay; using `torch.cuda.empty_cache()` after large evals, etc., can help if needed.

**Pseudocode Outline:**

Below is a condensed training loop pseudo-code incorporating many of the above steps:

```python
python# 1. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-7B-Chat")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-7B-Chat",
                                            quantization_config=bnb_config,
                                            device_map="auto")
model = prepare_model_for_kbit_training(model)  # prepare for LoRA + 4-bit

# 2. Configure LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                         lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # sanity-check that only LoRA params are trainable

# 3. Prepare data
train_data = load_dataset('json', data_files='train.jsonl')['train']
val_data = load_dataset('json', data_files='val.jsonl')['train']

# Function to format and tokenize one sample
def preprocess(sample):
    user_hist = sample['history']  # e.g., list of item titles
    candidate = sample['candidate']
    label = sample['label']  # 1 or 0, or "yes"/"no"
    # Build prompt text:
    prompt = f"User's purchases: {', '.join(user_hist)}. Candidate: {candidate}. Should we recommend this? Answer:"
    answer = " Yes" if sample['label'] == 1 else " No"
    # (Ensure there's a leading space before Yes/No so tokenizer treats it as separate token)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_ids = prompt_ids + answer_ids
    # Truncate:
    input_ids = input_ids[:512] 
    # Create labels - mask out prompt tokens:
    labels = [-100]*len(prompt_ids) + answer_ids 
    labels = labels[:512]
    return {"input_ids": input_ids, "labels": labels}

train_ds = train_data.map(preprocess, remove_columns=train_data.column_names, num_proc=4)
val_ds = val_data.map(preprocess, remove_columns=val_data.column_names, num_proc=4)

# 4. Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors='pt')

# 5. Define metrics
import evaluate
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # For generative models, preds are token ids; get final token of each prediction
    # Assuming Trainer generates preds via greedy decode:
    pred_labels = []
    for p in preds:
        # find first non -100 label in each label seq? Actually, we might customize evaluation to just generate "Yes"/"No"
        # For simplicity, say we decode:
        text = tokenizer.decode(p, skip_special_tokens=True)
        pred_labels.append(1 if "Yes" in text else 0)
    labels = [l for seq in labels for l in seq if l != -100]  # this gives all actual label tokens
    # Here labels should correspond to "Yes"/"No" tokens as well, we simplify:
    true_labels = [1 if tokenizer.decode([l]) == "Yes" else 0 for l in labels]
    result = accuracy.compute(predictions=pred_labels, references=true_labels)
    result.update(f1.compute(predictions=pred_labels, references=true_labels, average="binary"))
    return result

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="outputs/qwen3_rec_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-4,
    warmup_ratio=0.02,
    lr_scheduler_type="cosine",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    bf16=True,
    optim="paged_adamw_32bit",  # or 'adamw_bnb_8bit' if bitsandbytes is available
    max_grad_norm=0.5,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="wandb"
)
# 7. Initialize Trainer and train
trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
# 8. Save LoRA adapter and/or merged model
trainer.model.save_pretrained("outputs/qwen3_rec_lora_adapter")
```

This pseudocode covers the main components: loading model with 4-bit, applying LoRA, preparing data with masked labels, and training with evaluation. In practice, you may want to adjust how predictions are done – e.g., you might restrict evaluation to generating one token (to reliably map to yes/no). With the Trainer’s default generation, we might need to customize `Trainer.predict` to ensure it only generates the label token.

**Monitoring memory**: During training, use `nvidia-smi` or Torch logs to ensure usage is around, say, 20GB out of 24 (leaving some headroom). If it’s maxing out, reduce batch or seq length. If it’s too low, you might increase batch or consider using a larger model.

**Validation**: The trainer will evaluate every X steps. Make sure the eval doesn’t use too large a batch or sequence, as that could also OOM (the Trainer uses the same batch size for eval by default, which is fine if val set similar size examples).

**Training time**: A 7B model with LoRA on a 4090 can train quite fast. E.g., one report fine-tuned Qwen3-8B with 4k examples in ~30 minutes. For 50k examples, expect a couple of hours. For a million, perhaps ~1 day if fully utilized. Monitor throughput (tokens/sec).

In summary, the plan is to fine-tune Qwen3 similar to how one would fine-tune LLaMA or Mistral using LoRA, with adjustments for our recommendation prompt format. The above configuration is a solid starting point that should train on a 4090 without issues. Next, we focus on what to watch out for during training and how to ensure we get a good model.

## 4\. Key Engineering Tips and Hyperparameters to Watch

Before launching training, it’s helpful to have a **checklist of critical hyperparameters and techniques** specific to this LoRA + recommendation setup. These will affect model performance and generalization:

**Hyperparameters that matter:**

- **LoRA Rank (`r`) and Alpha:** As discussed, these control adapter capacity. Start with a small rank (8 or 16). A higher rank can capture more patterns but can also cause the model to deviate (catastrophic forgetting or overfitting). Monitor how training goes with rank 8; if underfitting, consider rank 16 with a proportional alpha increase. Alpha scaling should usually be left at default or a moderate value (e.g., 16 or 32). If alpha is too high relative to rank, the updates might dominate; too low and training might be slower. In practice, alpha = 16 works well for rank 8–16.
- **Target Modules:** double-check these. If for some reason Qwen3’s layer names differ, adjust accordingly. You want LoRA on at least the attention layers. If your model is MoE, you might also need to target expert layers (some code suggests LoRA for MoE requires stacking adapters for each expert). For dense Qwen, the list we gave is fine. Not tuning feed-forward layers (i.e., leaving out up\_proj/down\_proj) can preserve more of the base model’s original language modeling ability at the expense of some task-specific capacity. If you notice the model still retains unrelated chattiness or refuses (which you might if system prompt says “only Yes/No”), that’s fine in our case. If it were not learning the recommendation behavior strongly enough, adding LoRA to FFN (which we did by default) or increasing rank can help.
- **Learning Rate and Warmup:** This is often the most sensitive training hyperparam. Too high LR: model might converge fast to a suboptimal solution (e.g., always output “No”), or loss might fluctuate. Too low LR: slow learning or getting stuck. We chose 1e-4 with warmup 2%. After a few hundred steps, watch training loss: if it’s not dropping at all, maybe LR is too low; if it drops super fast and then plateaus or val loss starts rising quickly, could be too high. The warmup is important to avoid shocking the LoRA params (especially since the base model is initially good at language, we want a gentle transition into focusing on our task). If using a large dataset (millions of steps), consider a **cosine decay with many epochs**: it will reduce LR towards end which can improve final performance. For example, if training 3 epochs on a medium dataset, cosine schedule will anneal each epoch.
- **Batch size (and grad accumulation):** The *effective* batch (per update) can affect stability. If your batch is too small (e.g., 1 or 2), loss might be noisy and you might see the model oscillate (especially with higher LR). Too large a batch might smooth things but could also cause the model to predict the majority class predominantly if the data is imbalanced. Ensure you have a reasonable amount of both “yes” and “no” in each batch on average (random shuffling usually takes care of this). If your dataset is highly imbalanced, you may need to **stratify or up-sample** to avoid batches of all negatives, for instance.
- **Sequence length & truncation strategy:** Decide this upfront and implement consistently. If truncating to last N interactions, that’s effectively a hyperparam. E.g., using last 10 vs last 100 items can change model input distribution. We expect diminishing returns after a certain history length: including 50 purchases vs 100 might not significantly improve accuracy but will use more memory. As a rule, set a limit that covers ~95% of users’ histories. If 95% of users have ≤20 items, use 20; if some have huge histories, maybe 50. *Keep this constant during training* (don’t suddenly feed a much longer history at inference if model never saw that length during training). If you want to handle long histories, you must expose the model to them during training, which means enabling longer `max_length` and perhaps using gradient checkpointing.
- **Positive/Negative sample ratio:** In recommendation data, typically negatives far outnumber positives (a user doesn’t buy most products). If you naively label all unpurchased items as “no”, your training set will be extremely imbalanced. It’s common to **down-sample negatives** for training – e.g., for every positive example (user, item they actually bought), include perhaps 1 or 2 negative examples (user, random item they didn’t buy). Ensure your data preparation did this, or if not, use a loss function that can handle imbalance (like weighting positives more). We saw in the multi-label example, they computed label weights to give higher weight to rare classes. For binary, you could set a `pos_weight` in `BCEWithLogitsLoss` if using a manual training loop. If using Trainer with cross-entropy, you might artificially balance the dataset. **Action:** check the proportion of “yes” in your training data. If it’s, say, 5% yes, 95% no, the model could just predict “no” always and get 95% accuracy. To avoid that, either **balance the samples** (preferable) or use techniques like **focal loss** or weighting. A simple approach: for each positive, sample N negatives (maybe N=1 to 3) for that user. This yields roughly a 25-50% positive ratio in training, making the task about distinguishing rather than overwhelming with negatives.

**Regularization and generalization:**

- **Label smoothing:** This is sometimes used in classification to prevent the model from becoming too confident. For a two-label task, label smoothing might mean instead of hard 0/1 targets, use 0.9/0.1 or similar. Given our output is a single token prediction, we likely won’t use label smoothing here (it’s more common in multi-class scenarios with cross-entropy). Also, since “Yes” vs “No” should be a sharp decision, we actually want the model to be confident when it’s correct. Overconfidence is only an issue if the model overfits to training and can’t calibrate on new data. I’d skip label smoothing unless you see evidence of severe overfitting (e.g., near-perfect train accuracy but much lower val accuracy).
- **Dropout:** We already have LoRA dropout. The base model also has dropout in forward passes (most likely Qwen uses dropout in attention/ffn during training). The HF Trainer will enable dropout by default during training. Ensure `model.train()` is called (Trainer does this). If you see overfitting, you could increase dropout rate in LoRA or consider adding dropout to the input (e.g., randomly dropping some history items during training to make the model robust to missing interactions). For example, each epoch, for each user history, you might randomly omit one item 10% of the time. This could simulate noise and help generalization. It’s a form of data augmentation in recommendation – akin to leave-one-out training.
- **Freezing layers:** With LoRA, the base model is essentially frozen, so we don’t need to manually freeze anything else. That said, if you have a classification head (Method B2), you might choose to **freeze all base model weights including LoRA for a few initial steps** to train only the new classification head, then unfreeze LoRA. But since we integrate LoRA from the start and we want the adapter to do the heavy lifting, this isn’t necessary. (This technique is more for full fine-tuning scenarios or when adding a new final layer.)
- **Cold-start handling:** In recommendations, a known issue is new users or new items that weren’t in training. Our LLM approach doesn’t have an ID embedding matrix like traditional recsys, so it’s a bit different: if a new item appears (ID not seen), the model can’t directly handle it unless it’s presented with some description. **Mitigation:** Represent items with textual features (name, category) so the model can generalize to new ones by semantics. For new users with no history, decide on a convention: e.g., history could be “<none>” or simply an empty list. The model should learn that with no history, maybe default to “No” (or maybe “Yes” if item is very generally appealing? But it won’t know that without info). In training, include some examples of users with empty or very short history, so the model sees such cases. That’s data preparation more than hyperparam, but it’s crucial for generalization to new users.
- **Data shuffling:** Always shuffle your training data well (HF Trainer does by default each epoch). Since data might be ordered by user or time, shuffling prevents temporal or user-order biases from affecting training order.
- **Continual adaptation vs full re-training:** If later you get more data, LoRA allows you to fine-tune further without full retraining. Just a note: keep an eye on whether multiple fine-tuning rounds degrade performance; sometimes it’s better to merge LoRA weights into the base and then start a new LoRA for additional fine-tuning.

**Evaluation metrics:** We should evaluate the model on both the binary classification level and a simulated recommendation ranking scenario:

- **Binary classification metrics:**
	- **Accuracy**: straightforward, how often did it predict correctly yes/no.
	- **Precision & Recall**: especially if classes are imbalanced. For example, if “yes” is the rare positive class, we care about recall (how many of the actual yes’s did we catch) and precision (of the items we predicted yes, how many were truly yes). We can combine into **F1-score**.
	- **ROC AUC:** since we can get a probability (the model’s confidence in yes vs no), AUC is a threshold-independent measure of ranking quality for the binary task. If using generative output, we can derive a score: e.g., take the log-probability the model assigns to " Yes" vs " No" when fed the prompt. If using a classifier head, softmax score for "yes" is directly available. AUC is useful if we want to choose a threshold or use the model in different sensitivity settings. It’s a good indicator of overall discriminative ability.
	- **Log loss / cross-entropy:** not usually reported, but if you want to calibrate the probabilities, log loss on val can help fine-tune regularization.
- **Top-K recommendation metrics:** Ultimately, if this model were deployed in a recommender, we’d use it to rank multiple candidates and pick the top K with “Yes”. To evaluate that, for each user in validation, you need a set of candidate items (including the ones they actually interacted with as positives and some negatives).
	- **Hit Rate@K (or Recall@K):** Does the model’s top K include the actual next item the user interacted with? If you have one ground-truth item per user (like next purchase), and you score N candidates, then HR@K is 1 if the ground-truth is in the top K, 0 if not (averaged over users).
	- **NDCG@K:** This accounts for rank position of the hit. If we only have one relevant item per user, NDCG@K reduces to a discount factor if that item is in the rank. But if you have graded relevance or multiple positives, NDCG is more general.
	- **MRR (Mean Reciprocal Rank):** The reciprocal of the rank of the true positive. Useful if exactly one true item per user.

These require a different evaluation setup than just feeding one (user, item) pair at a time. Typically, you’d:  
\* For each user in the test set, take their actual next purchase as the positive. Sample a bunch of negatives (items not purchased by them) to form a candidate set (like 100 or 1000 negatives).  
\* For each user, feed the model the prompt with each candidate and get a score (the probability of “Yes” or the logit).  
\* Rank the candidates by score and compute if the positive is in top K.

This is computationally heavy if done naively for many users and items. On a single GPU, it’s doable if you restrict negatives (e.g., 100 per user) and batch the scoring. Since our model is relatively slow compared to specialized recsys models, a common approach is to use a **two-stage eval**: use a simpler model to pre-select top 100 candidates per user, then apply the LLM to re-rank those 100. This aligns with realistic systems where LLM is the re-ranker. Then compute top-K metrics on that list.

To compute these efficiently:  
\- Use vectorization: group prompts into batches and use the model’s `generate` or `predict` in batches. E.g., batch of 32 prompts at a time to utilize GPU.  
\- If using the classification head approach, you can score many items in parallel since it’s just one forward pass giving logits for each item in the batch.  
\- If using the generative approach, you might use the logits of the first generated token for yes/no as the score. With HuggingFace, you can get the logit output by doing `model(**inputs)` where inputs are tokenized prompt (no labels), and look at `outputs.logits` for the position where the answer token would be. For example, take the last token of prompt (the position of "Answer:" token) + 1, and take the logit for the "Yes" token at that position. Compare to "No" token logit. This requires careful indexing but avoids full generation for each candidate.

We definitely want to monitor AUC and F1 during training as proxies for ranking quality. For top-K, that’s more for offline evaluation after training (you can do it on a subset of data due to expense).

**Tips from real-world fine-tunings:**

- A small LoRA rank with a slightly higher learning rate was a combination found to work well for Qwen3 – it preserved base knowledge and still adapted to the task. In other words, don’t be afraid to use e.g. 2e-4 if using rank 8, as long as you watch metrics. The LoRA Without Regret guide suggests LoRA LR in 1e-4 to 5e-4 range can be optimal.
- **Overfitting signs:** If you notice validation accuracy starts dropping while train keeps rising, or the model’s outputs become degenerate (e.g. always “Yes” regardless of input), that’s a clear overfit or imbalance issue. Mitigations: increase dropout, reduce rank, add regularization, or simply stop training earlier. Also check if your negative sampling strategy might be too easy (e.g., random negatives that are obviously not related to user’s interest – the model might quickly learn a trivial rule and then struggle to learn subtler differences, leading to flatlining val metrics).
- Some practitioners mention **freezing the embedding layer** or first few transformer layers in full fine-tuning to avoid catastrophic forgetting. In LoRA, base embeddings are untouched anyway, so not an issue. But if you added any new tokens (for product IDs, for instance), ensure you initialize them properly (maybe average of known tokens) – but again, using titles avoids needing new vocab tokens.
- **Avoid extremely long histories in training:** It’s observed that forcing an LLM to attend to very long sequences of mostly irrelevant info can hurt its learning – it might waste capacity modeling positional dependencies that aren’t needed. If you have some users with thousands of interactions, you might actually *hurt* training by always feeding those in full length. It’s better to truncate or split those into multiple training examples (though splitting breaks the concept of “next item” label unless done carefully). Simpler: truncate to a max length. The base Qwen can handle long input, but if it never needed to utilize that for the task, it could confuse it. So feed what’s likely to be relevant (this ties back to truncation strategy).
- **Mix of data:** If you had multiple datasets or domains, sometimes fine-tuners mix them and use **task-specific tokens or prompts** to help model distinguish. For example, if you combined MovieLens (movie recommendations) and an e-commerce dataset, you could add a domain token in prompt like “\[Domain: Movies\] User history:...”. This can improve generalization across domains. In our case, probably focusing on one domain is fine.
- **Failure modes:**
	- The model might lean towards always predicting the majority class (“no”) if imbalance is not addressed. You’ll notice high accuracy but low recall for yes. The fix is data balancing or threshold tuning.
	- The model might **memorize popular items** – e.g., always say “Yes” for certain popular products regardless of user. If in training data many users who had some items did end up buying a very popular item, it may pick up bias “if candidate is iPhone and user bought anything tech, answer Yes.” This might not generalize to a user who doesn’t actually want an iPhone. Monitoring item-specific error patterns could reveal this. If needed, you could introduce some noise (e.g., include some counter-examples where a popular item is given but user history suggests a different brand loyalty).
	- Another mode: model outputs something off-format like “Yes, I would recommend it.” This means it didn’t strictly follow our one-word requirement. Fix by editing the prompt format (explicitly say “Yes or No only”) and possibly by checking some outputs during training (you can adjust the format or do a second stage of training with a few epochs of strictly format-correct data).
	- If using chat format, sometimes the model might output `<|assistant|>Yes` or include role tokens if the decoding isn’t handled properly. Make sure to use the Qwen chat API or process output to get the content. During training, if we format the data as shown, this shouldn’t happen because the target is only “Yes”.

**Regular check**: Keep an eye on **gradients and losses**. If you see loss becoming NaN or gradients exploding (rare with LoRA), you might reduce LR or use gradient clipping (we already set that). If training loss is decreasing but val loss is not, likely overfit – consider early stopping or more regularization.

In summary, pay special attention to **LoRA rank, learning rate, batch size, and data balance** as these will largely determine success. Also ensure your metric tracking is comprehensive – it’s easy to get a model that looks great on accuracy but actually just predicts “no” always; hence use precision/recall or AUC to really judge it.

Below is a concise **checklist** of key points before training:

- **LoRA params:** Set `r = 8–16`, `alpha ~ 2*r`, `dropout ~0.05`, `bias='none'`. Target Q, V (at least) and ideally K, O, and FFN layers.
- **Quantization:** Use 4-bit or 8-bit to fit model in memory. Use `bf16` for computation if available.
- **Optimizer & LR:** AdamW (8-bit if possible). LR ~1e-4 (monitor and adjust), warmup ~100–500 steps.
- **Batch/Seq:** Effective batch ~32. Truncate history to max length (decide e.g. 256 tokens). Shuffle data.
- **Balance classes:** Ensure training data isn’t overwhelmingly one class. Use negative sampling or weighting to get a reasonable mix.
- **Metrics:** Plan to track accuracy, F1, AUC at minimum. Possibly track the ratio of “yes” predictions to catch bias.
- **Regularization:** LoRA dropout on. Maybe drop some history entries randomly to augment. No label smoothing unless needed.
- **Monitoring outputs:** During training, periodically feed a few known examples (small script or use `Trainer.callback` to generate on some fixed prompts) to see if outputs make sense (e.g., not always “no”).
- **Stop criteria:** Decide on early stopping metric (e.g., if val AUC doesn’t improve for X evals or starts dropping).
- **Evaluation:** After training, evaluate on a hold-out set with top-K metrics to ensure the model’s recommendations align with expectations.

Having this checklist and the references above will help maintain focus on what hyperparameters and settings to tune for optimal results.

## 5\. Monitoring and Debugging During Training

When fine-tuning the model, it’s important to monitor both **quantitative metrics** and **qualitative outputs** in real-time. We’ll set up logging (preferably with a tool like Weights & Biases or TensorBoard) to catch issues early and to decide when to stop training. We should monitor model performance, training behavior, and resource usage.

### Metrics to Track During Training

**Training vs Validation Loss:** Log the training loss every few updates (e.g., every 50 steps) and validation loss at least every epoch or every N steps (like every 500 steps). A steadily decreasing training loss is expected. **Compare it to validation loss**:

- If training loss keeps dropping but validation loss bottoms out and then starts rising, that’s a sign of **overfitting**. You might observe this after many steps: training loss approaching 0.0 while val loss increases – at that point, stop training (or consider earlier stopping to save the model at the best epoch).
- If both training and validation loss plateau at a high value and metrics are low, the model is **underfitting** (or the task is very hard or data insufficient). Underfitting might mean we need more training steps, higher model capacity (maybe increase LoRA rank or even choose a larger base model), or a higher learning rate. It could also hint at mis-formatted input – if the model can’t lower loss, maybe it’s confused by the prompt format or the label token.

**Classification Metrics:** We will log:

- **Accuracy** and **F1** on the validation set every evaluation interval. Accuracy is easy but can be misleading if classes are imbalanced. F1 (for “yes” as positive class) gives a balance between precision and recall.
- **Precision & Recall** explicitly, or at least F1 as above. If we find the model tends to predict mostly “no”, we’ll see high precision on “no” but low recall on “yes” or a low F1.
- **AUC (ROC AUC):** This is a great metric to track because it’s insensitive to threshold. Even if the model is biased to saying "no", as long as it ranks positives higher on average, AUC will reflect that. We want AUC to improve steadily. A random guess would be 0.5 AUC, so anything above that shows learning. If AUC plateaus and doesn’t improve for a while, the model may have converged.

For example, in W&B or TensorBoard, have a chart of **Val AUC vs. Steps** and **Val F1 vs. Steps**. If they flatten out or start dropping, consider early stopping.

**Class Imbalance Monitoring:** It’s useful to log the **distribution of predictions**. For instance, every evaluation, calculate the fraction of examples where the model predicted “yes”. If that fraction is trending to 0 or 1 exclusively, that’s a red flag:

- If it goes to 0% yes predicted, model became too negative-focused.
- If 100% yes, it’s the opposite.  
	Ideally, if your validation set has X% positives, the model’s predicted positives should at least be in that ballpark. We can log “Predicted\_Positive\_Rate” on val. Similarly, log the **true positive rate and true negative rate** (or confusion matrix counts). This helps spot if model is only doing one thing.

**Steps to detect issues:**

- **Overfitting:** You’ll see a divergence between train and val loss: e.g., train loss keeps going down, val loss starts going up. Also val accuracy/AUC starts decreasing after a certain point. To detect early, one can set `load_best_model_at_end=True` and `evaluation_strategy="steps"` so that you always keep track of the best validation score. W&B can highlight the best epoch.
- **Underfitting:** If after one epoch or a good number of steps, training loss is not far below validation loss and both are relatively high (compared to what a baseline might be), the model might be underfitting. This could mean we need more epochs or adjustments. For instance, if after training, accuracy is barely above random (like 60% on a balanced set), we underfit.
- **Label imbalance problems:** The predicted positive rate above is key. If you see, for example, the model predicts “yes” only 5% of the time but in the data 20% are yes, it’s biased. This might show up as high overall accuracy (because it’s getting all negatives right) but low recall. The **PR curve** or at least precision/recall values should be examined. W&B can plot a confusion matrix if you log it (true vs predicted counts).

You might also log **loss for yes and no separately** if you can (or at least track metrics for each class). If one class’s loss is much higher, the model struggles with that.

### Qualitative Monitoring (Example Predictions)

Numbers alone can sometimes hide what’s going wrong. It’s extremely useful to periodically **print or log example model outputs**:

- Choose a handful of representative users from the validation set. Perhaps a mix: one with a clear preference pattern, one with an out-of-distribution history, one that is a tricky edge case.
- For each (user, candidate, true label), run the model and get its output (Yes/No).
- Log these in a table or text panel in your dashboard. In W&B, for instance, you can use `wandb.log({"examples": wandb.Table(data=[...])})` or log text outputs.

Over time (say every epoch), see how these predictions change:

- Early on, the model might just output always “No” (or even gibberish if not formatted right).
- As training progresses, you should see it start to predict “Yes” for some and matching the ground truth more often.
- If later it starts overfitting, you might see it output “Yes” for almost everything or some weird pattern.

Logging the **model’s raw logits** for those examples can also help debug. For example, maybe it’s always slightly favoring "No" with probability 0.6 vs 0.4 for "Yes". If we see that, we might adjust the classification threshold after training.

Also consider logging a few **failure cases**: after training, identify some users where the model was wrong, and inspect. Perhaps the user’s history had items that the model doesn’t understand relate to the candidate (maybe due to text or because it needed more world knowledge). This could inspire adding features (like categories) or improving the prompt.

**Comparison to ground truth:** We can have our logging output like:

From such lines, we might notice if the model is making an obvious mistake (“Yes” to recommending a Samsung phone to a hardcore Apple user – which might indicate it just sees electronics and says yes, missing the brand loyalty concept). This qualitative insight is valuable.

**Tracking evolution:** If using W&B, you can create a custom plot of F1/accuracy over time and even set alerts (e.g., ping if val loss starts increasing). Manually, you can just eyeball it.

### Resource and Throughput Monitoring

To ensure we’re utilizing the 4090 effectively and avoid crashes:

- Monitor **GPU memory usage:**`nvidia-smi` will show memory consumption. Ideally, you want to use most of the 24 GB but not 100%. If it’s at 23900 MiB (very close to 24 GB), you risk OOM if any spike happens. Perhaps reduce batch by 1 to give headroom. If it’s only using 12 GB, you’re under-utilizing – you might increase batch or try a larger model or context.
- Monitor **GPU utilization (%)**: If using accelerate or W&B, they might log GPU utilization. You want it high (~90%+) during training. If it’s low, maybe the data loading is a bottleneck. Our dataset is not huge in terms of I/O (text data), but if your CPU is slow at tokenization, consider using multiple preprocessing workers (`Dataset.map(num_proc=4)` we did).
- **Throughput (tokens/sec or samples/sec):** The Trainer logs `samples/sec`. Keep an eye on it when adjusting things. For example, with gradient checkpointing on, you might see a drop in throughput (maybe 20-30%). That’s expected. If it’s extremely slow (< 1 step/sec for small batch), something’s off (maybe too high overhead logging or not using GPU fully). On a 4090, fine-tuning 7B with batch ~8 should yield on the order of dozens of samples/sec (depending on seq length). You can also compute tokens/sec = samples/sec \* (average seq length). This helps to estimate training time for X steps.
- If using W&B, you can log **system metrics** (there’s an integration to log GPU utilization, CPU, etc., automatically with `wandb.watch` or their system monitor).
- **Temperature of GPU:** not usually an issue with proper cooling, but long runs at 100% might thermal throttle if cooling is inadequate. Rare on desktops with good cooling, but worth checking if suddenly training slows down.

**Handling OOMs:** If you get an Out-Of-Memory error:

- The training loop will likely crash. Check the log to see if it was during forward or backward. If it’s during backward on the first step, you clearly exceeded memory. Lower batch or seq.
- If it happens after several steps (maybe during evaluation when the eval batch was larger or sequence un-padded and suddenly long), it could be a spiky input causing it. For safety, enforce the max\_length strictly (clip inputs). Also set `per_device_eval_batch_size` possibly smaller if val data has variability.
- Sometimes memory fragmentation leads to an OOM after many steps. PyTorch usually manages this, but if it occurs, try setting `gradient_checkpointing=True` or even offloading some parts to CPU (Accelerate can offload optimizer states).
- The order of options to try: **reduce batch size** > shorten sequence > gradient checkpoint > use more aggressive quantization (if not already 4-bit).

**Healthy ranges:**

- **Training loss**: starting from maybe ~0.7–1.0 (if classes balanced, that’s like logloss of a random guess) and ideally decreasing towards 0.2 or lower.
- **Accuracy**: if balanced data, random is 50%. We’d like to see it go towards maybe 90%+. If data is imbalanced (say 10% yes), accuracy might start high (if model just says "no") but F1 will be low. So watch F1: a good model might get F1 0.8 or above depending on data difficulty.
- **AUC**: random 0.5, a decent model 0.8+, an excellent one 0.95+. The Hemang project claimed 98% accuracy which implies near perfect AUC as well, but that might be on a filtered dataset.

**Utilizing W&B or TensorBoard:**

- In W&B, set `WANDB_PROJECT="Qwen3-Recommender"` and `report_to="wandb"` in TrainingArguments. This will automatically log losses and metrics.
- You can log custom things: e.g., in `compute_metrics`, also log confusion matrix or class distribution. W&B allows `wandb.sklearn.plot_confusion_matrix`.
- Create custom **alerts** in W&B: e.g., alert if val loss increases 3 evals in a row (possible plateau/overfit).
- For qualitative, use `wandb.Table` as mentioned or `wandb.log({"examples": text})` where text is e.g. a multi-line string of example predictions.

**Examples of graphs to check:**

- *Training & Validation Loss vs. Steps:* should see train loss > val loss initially (because val is on easier terms, the model is still mostly base so it might do okay on val if baseline is good), then as training goes on, train loss will dip below val. We want to stop when they diverge.
- *Accuracy/F1 vs. Steps:* should increase then possibly level out. If you see it drop, investigate data or possible overfit.
- *AUC vs. Steps:* nice to see a smooth upward curve. If it peaks and then declines, again likely overfitting.
- *Predicted Positive Rate vs. Steps:* might start low, then hopefully converge to roughly actual positive rate if calibrated. If it goes to extremes, know that the model might be biasing.

**Debugging during training:**

- If metrics aren’t improving at all in the first epoch, something could be wrong (e.g., the prompt format might be misaligned with how we apply labels). In that case, print out a batch’s input IDs and labels to make sure the labels align with "Yes"/"No" tokens. Ensure the tokenizer encoding is as expected (especially if we needed a space before "Yes" to tokenize as one token).
- If training diverges (loss goes NaN): that’s rare with LoRA at reasonable LR. But if it happens, lower LR by 10x and retry from the last checkpoint or from scratch. Also check if any data has something odd (like extremely long input).
- Keep an eye on **gradient norms** if possible. If they explode, you’ll see NaNs. Our grad clip should prevent that.

**After training:** Use the monitoring logs to decide the best checkpoint (if `load_best_model_at_end=True`, Trainer will do that). Save the final adapter (which Trainer does if best at end). You can then do a test evaluation and log those results too.

In summary, treat the training process as an experiment you’re actively watching:

- Use real-time metric plots to know when to stop or adjust.
- Inspect example outputs to catch format issues or biases.
- Watch GPU usage to ensure efficiency and no crashes.

By having this monitoring in place, you’ll have a much smoother fine-tuning experience and a higher likelihood of catching issues early (rather than finding out at the end that the model just outputs “no” for everything!).

## 6\. End-to-End Recipe & Prioritized Action Plan

Finally, let’s summarize an actionable recipe to go from data to a deployed model, and list concrete next steps in order of priority.

### End-to-End Fine-Tuning Recipe

**Data Preparation:**

1. **Gather & Clean Data:** Use your main dataset (e.g., MovieLens, Amazon, or DeliveryHero) of user purchase histories. Ensure you have a clear list of items per user and known positive interactions.
2. **Negative Sampling:** For each user and each positive interaction (purchase), sample one or more negative items (items the user did not buy) to create training pairs. Aim for a balanced or mildly imbalanced dataset (e.g., 1:1 or 1:2 ratio of yes:no labels).
3. **Format Data to JSONL:** Decide on prompt format (chat or instruction style). Create a JSONL file where each line is a training example with the formatted prompt and the label. For example, `{"instruction": "...","input": "...","output": "Yes"}` for each instance, or the ChatML format JSON as shown earlier. Include some examples with empty history if applicable.
4. **Tokenization Check:** Test the tokenizer on a sample prompt to ensure “Yes” and “No” are tokenized properly (and not broken into subwords unexpectedly). Adjust prompt spacing or add to tokenizer vocab if needed.

**Model & Environment Setup:**  
5\. **Select Qwen Model:** Choose `Qwen/Qwen3-7B-Chat` (for chat format) or `Qwen/Qwen3-7B` (for base). If using a larger model like 14B, ensure you have bitsandbytes installed for quantization.  
6\. **Set up Environment:** Install required libs: `transformers`, `peft`, `accelerate`, `datasets`, `bitsandbytes`. Also install `wandb` for logging. Log in to Hugging Face Hub if needed to download model.  
7\. **Load Model in 4-bit:** Load the Qwen model with `AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True, device_map="auto")`. Load the tokenizer (use `AutoTokenizer` and include `trust_remote_code=True` if Qwen uses custom code).  
8\. **Apply LoRA:** Use `peft.LoraConfig` with the chosen hyperparams (r, alpha, modules). Wrap the model with `get_peft_model`. (Optional: `print_trainable_parameters()` to confirm the count is small.)  
9\. **Prepare Trainer:** Define `TrainingArguments` for 1-3 epochs, batch size, LR, etc. Enable bf16, gradient checkpointing if needed, and logging steps. Set `evaluation_strategy="steps"` and `save_strategy="steps"` to evaluate periodically and save checkpoints.

**Training Loop:**  
10\. **Data Collation:** Create a data collator for padding. For causal LM, ensure labels are correctly aligned (as described in section 3).  
11\. **Train Model:** Instantiate `Trainer` with model, data, collator, and metrics. Start training with `trainer.train()`. The Trainer will handle logging to console and W&B.  
12\. **Monitor:** Keep the W&B or console open to watch losses and metrics. Ensure GPU is being utilized. If something looks wrong (e.g., loss not changing), pause/abort and debug (check data pipeline).  
13\. **Evaluate:** After training, the best model (with lowest val loss or highest chosen metric if configured) will be loaded. Run `trainer.evaluate()` on a test split to get final accuracy/AUC. Also, perform the top-K evaluation offline if needed by scoring a sample of users.

**Model Saving & Deployment:**  
14\. **Save LoRA Weights:** Use `PeftModel.save_pretrained` to save the LoRA adapter. Also save the tokenizer (tokenizer is same as base).  
15\. **(Optional) Merge LoRA:** If you want a standalone model, you can merge the LoRA into the base using `PeftModel.merge_and_unload()`. This gives you a regular model with weights modified by LoRA (useful if you want to deploy without PEFT).  
16\. **Test Inference:** Load the model (base + LoRA or merged) in a script. Feed a few example prompts and confirm it outputs "Yes"/"No" as expected. Use `model.generate` or `model(**inputs)` to get logits. Possibly restrict generation to just one token.  
17\. **Deploy/Integrate:** Depending on usage, you might deploy this model behind an API. Ensure to send prompts in the exact format as training. For example, if using chat format, include the system and user prompt structure.

**Iteration:**  
18\. **Analyze Results:** Check metrics. If performance is not satisfactory, consider next steps: more data, higher model size, tuning hyperparams. Identify where the model makes mistakes (maybe item cold-start or specific categories).  
19\. **Hyperparameter Sweep:** You might experiment with different LRs (e.g., 5e-5 vs 2e-4) or LoRA ranks (8 vs 16) on a subset of data to see effect. Do this systematically if time permits.  
20\. **Repeat Fine-tune if Needed:** Apply any changes (like new prompt format or more balanced data) and retrain. LoRA training is cheap enough to iterate multiple times.

This recipe gets you from raw data to a fine-tuned Qwen3 that can recommend yes/no.

### Prioritized Action List

To efficiently execute this project, here’s a prioritized to-do list:

**Step 1: Review Reference Implementations**

- **Clone and Read Qwen LoRA Examples:** Start by examining the official Qwen SFT example. Although it’s geared to chat instruction tuning, note how they handle data and LoRA integration. Also skim the HuggingFace Optimum script for Qwen3 (finetune\_qwen3.py) – focus on how they format the dataset and apply LoRA.
- **Clone LLM-RecSys Repository:** (`HEMANGANI/LLM-Recommendation-Systems`). Open the `README.md` and any code notebooks. Pay attention to their prompt engineering (the three-part format) and training workflow (they likely use Unsloth + LoRA). This will provide a concrete example of adapting LLMs to recommendation.
- **Read the Mistral fine-tuning blog code:** If available, check the linked repository in the HF blog (the author provided a link to code). Focus on how they use `AutoModelForSequenceClassification` with LoRA. This is instructive in case you want to try the classification head approach.

**Step 2: Set Up Minimal Training Script**

- Using the references, start a script or notebook for fine-tuning Qwen3 on a small sample of your data. Implement data loading, model loading, LoRA config, and one training epoch. This “minimal training script” could be as we described in pseudocode. Aim to get it running end-to-end on, say, 1000 samples as a smoke test.
- Ensure logging to console or W&B is working. Check that after a few steps the loss decreases.
- This step is about getting a working pipeline before scaling up. It’s easier to debug issues on a tiny sample.

**Step 3: Hyperparameter Planning (Sweeps)**

- Decide on initial hyperparams (likely: r=8, LR=1e-4, batch=8, epochs=1 or 2). Plan a potential sweep: for instance, prepare configurations for LR = 5e-5, 1e-4, 2e-4; or r = 8 vs 16. **Priority:** first get a baseline training done with default guess (1e-4, r8). Only after seeing that result, do a formal sweep if needed.
- Configure W&B sweep or simply do manual runs one by one. Sweeps are optional; manual experimentation might suffice given time.

**Step 4: Monitoring Dashboard Setup**

- If using Weights & Biases, initialize a project now. Confirm that metrics like loss and accuracy are logging. Set up any custom charts (like a custom chart for AUC if not auto-logged).
- If sticking to TensorBoard, ensure `TrainingArguments.logging_dir` is set and run `tensorboard --logdir <dir>` to monitor.
- Set up alerting if possible: W&B allows email/slack alerts if a metric goes out of bounds. For example, alert if val accuracy drops by more than X.
- On the resource side, maybe use `nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --loop-ms=5000` in a separate terminal to watch GPU usage periodically. Or use a tool like `gpustat`. Not critical, but helps to ensure we utilize the GPU fully.

**Step 5: Full Training Run**

- Once everything looks good on a small scale, run the full training on the entire dataset. Monitor via the dashboard. This may run for hours, so periodically check in (or rely on alerts).
- After completion, note the final metrics. Save the final model and possibly push it to Hugging Face Hub (so you have a versioned copy).

**Step 6: Evaluation and Iteration**

- Evaluate on test set with both the classification metrics and a sample top-K as described. If results are satisfactory (e.g., AUC and accuracy meet your goals), you might finalize the model.
- If not, identify issues (e.g., poor recall for “yes”). Adjust accordingly: maybe the prompt format needs to emphasize certain aspects (“Recommend or not recommend?” vs “Yes/No”), or maybe the dataset needs more negatives of a certain type.
- Iterate: e.g., if model is too biased to “no”, try increasing positive sampling or adjust loss. Retrain LoRA (quick since base model stays same).
- Possibly try a larger model (if 7B underperforms and you have headroom, move to 14B with QLoRA).

**Step 7: Deployment**

- Integrate the model into your application. If using it as a re-ranker, you’ll feed it user histories and candidate and get yes/no out. Make sure to include the system prompt if required and strip the output properly (just check if “Yes” in output).
- Use the model’s probability (like softmax output or logits gap) as a score for ranking multiple items.
- Monitor the model in production – e.g., on new data does it still perform? If data drifts, you might need to periodically fine-tune on new data (LoRA makes this easy; you can even train new LoRAs and combine them if needed).

**Additional resources to explore (if time):**

- The Reddit discussion for insights on Qwen3 LoRA quirks.
- Unsloth’s documentation on fine-tuning (could have tips on memory or speed).
- “Awesome-LLM-for-RecSys” survey for any ideas on prompt engineering or known challenges with LLMs in RecSys.

By following this plan, you’ll leverage existing examples to quickly stand up your training, then iteratively refine hyperparameters and data formatting with close monitoring. The result will be a Qwen3 model finely tuned to output high-quality recommendations (“yes” or “no”) given a user’s history and a candidate product. Good luck with your fine-tuning!

**Sources:**

1. Qwen official fine-tuning documentation – data format and training script options
2. Reddit report on Qwen3 LoRA fine-tuning – optimal rank (8) and hyperparam settings
3. HF Blog (Lukas et al.) on LoRA classification – LoRA config for Mistral-7B (r=16, dropout 0.05, target attn only)
4. Hemang’s LLM-RecSys project – prompt strategy and achieved accuracy with LoRA on recsys data
5. Unsloth Qwen3 fine-tuning blog – memory optimization claims (70% VRAM reduction, flash-attention for long context)
6. Optimum Neuron Qwen3 example – LoRA configuration (r=64, α=128) and use of flash-attn and long context
7. “Practical Qwen3 LoRA guide” (Ivan’s Medium) – recommended LoRA defaults (r=16, α=16) and using KL to preserve base behavior
8. Mistral multi-label tutorial – training loop details, custom Trainer for weighted loss, etc., confirming viability on single GPU

---

1\. i see
```
Recommendation: We can actually combine the benefits: use a prompt style input (with clear instruction) but constrain the output to yes/no by design. The Format A examples above do exactly that – the question explicitly asks for “Yes or No” and the assistant answer is one word. This way we maintain an instruct prompt (which the Qwen model is used to) yet get a simple output space.
```
. can you give the format and example? 2. `1.8M * 2 (since each weight has grad and optimizer states in FP16) ~7.2M floats` how this calculation work? 3. what is gradient accumulation? will batch size affect training performance? given total epoch and training data keep the same. can you answer these questions seperately. no need to update original deep research document you created

---

Absolutely, let me answer each of your questions one at a time:

---

### 1\. Combined Format: Prompt-Style Input With Constrained Output (Yes/No)

This format blends natural language prompts (as chat or instruction-style) with an output that's **constrained** to either `Yes` or `No`. It benefits from instruction-following capabilities of Qwen3 but simplifies the label space for stable LoRA fine-tuning.

#### Format (ChatML Style)

#### Format (Instruction/Input/Output JSONL)

You can tokenize this into `prompt + label`, and supervise the model to predict only the `“Yes”` or `“No”` token.

---

### 2\. LoRA Weight Memory Breakdown

Here's how that estimate works:

- Assume ~1.8M total trainable LoRA parameters (from small rank and few target layers)
- For each trainable parameter:
	- You store the **parameter itself** (weights)
	- You also store its **gradient**
	- You also store **optimizer state** (e.g., Adam stores 2 additional values: mean and variance)

#### Total floats per weight:

- Weight: 1 float (FP16 or BF16)
- Grad: 1 float
- Optimizer: 2 floats (AdamW)

→ Total = **4 floats per weight**

So:

- 1.8M weights × 4 = 7.2M floats
- 7.2M floats × 2 bytes (FP16) = ~14.4 MB GPU memory

This is small and makes LoRA extremely efficient to train even large LLMs.

---

### 3\. What Is Gradient Accumulation? How Does Batch Size Affect Training?

**Gradient Accumulation** is a strategy where:

- You split a large logical batch into smaller microbatches (fitting in GPU memory)
- Instead of updating model weights after every microbatch, you **accumulate gradients** over N steps, then do one optimizer step

#### Why Use It?

- It simulates large batch training **without needing huge VRAM**
- Especially useful when using 4-bit LLMs with large context windows or limited GPUs

#### Example:

If you want an **effective batch size of 64**, but can only fit **8 samples per GPU step**:

- Set `per_device_train_batch_size = 8`
- Set `gradient_accumulation_steps = 8`

→ Accumulate 8 mini-steps before doing 1 weight update

#### Does Batch Size Affect Performance (Assuming Epochs + Data = Constant)?

Yes. Here's how:

| Factor | Larger Batch | Smaller Batch |
| --- | --- | --- |
| Updates per epoch | Fewer | More |
| Gradient noise | Lower (smoother) | Higher (noisier) |
| Convergence | May be slower (sharp minima) | May generalize better |
| VRAM use | Higher | Lower |

Even if total epochs and data are the same, **effective batch size** changes training dynamics:

- **Smaller batch + more steps** can **regularize better**, improving generalization
- **Larger batch** trains faster per epoch, but can overfit or plateau early

In practice, start with the **largest batch size you can fit**, and **tune learning rate accordingly**.

Let me know if you'd like to see a visual example of how batch + accumulation work!