# ThinkRec LoRA Reuse Plan

## 1. Target Fine-Tuning Data Shape

ThinkRec interleaves two training modes:
- **Recommendation classification (v2/v3)**: Prompt ends with a binary answer (`Yes`/`No` or concrete item). Only the logit of the positive token contributes to the loss.
- **Reasoning generation (v1)**: Prompt expects the model to emit a full preference explanation beginning with the binary verdict, and the full sequence is trained with causal LM loss.

To reproduce this behaviour we need a sample structure that provides, per interaction:
- `UserID` (int) and potentially session tokens for sequence models (e.g. SASRec).
- `HisItemList` / `HisItemTitleList`: ranked history, ideally both ids and natural language titles to let prompts reflect collaborative and content views.
- `TargetItemID` (+ optional `TargetItemTitle`).
- `label`: binary target for recommendation.
- `reason` (optional): textual rationale linked to the interaction.
- `User_emb`, `TargetItem_emb`, optionally `HisItem_emb_list`: tensors produced by the collaborative model (MF/SASRec/etc.) used to replace placeholders in the prompt.

A minimal JSONL example aligned with ThinkRec expectations:
```json
{
  "UserID": 123,
  "HisItemIDs": [12, 45, 73, 91],
  "HisItemTitles": ["Space Opera", "Time Dilation", "Alien Diplomacy", "Quantum Drift"],
  "TargetItemID": 155,
  "TargetItemTitle": "Graviton Uprising",
  "label": 1,
  "reason": "Yes. The user gravitates to hard sci-fi with political intrigue; the target shares the same mix of military strategy and physics puzzles.",
  "User_emb": [...],
  "TargetItem_emb": [...],
  "History_embs": [[...], ...]
}
```
We can store embeddings as base64/npz for offline datasets; during training, a preprocessing collate step loads tensors and injects them into prompts.

## 2. ThinkRec Components Worth Reusing

| Component | Location | Reuse Rationale |
|-----------|----------|-----------------|
| Prompt/embedding injection utilities (`prompt_based_encode_v3`, `recprompt_wrap_v2`, `encode_recdata_v2`) | `previousWorks/ThinkRec/minigpt4/models/minigpt4rec_v3.py` | Provides the mechanics for replacing prompt placeholders with LoRA-projected collaborative embeddings, handling token alignment and padding. |
| Projection heads (`self.llama_proj`) and projection pipeline | same file | Projects MF/SASRec embeddings into language-model hidden space; can be refactored into a standalone module to convert our recommender features into token embeddings. |
| LoRA configuration + multi-adapter handling (`set_user2group`, `adaptive_lora`) | same file | Offers a template for dynamic adapter selection based on clustered user representations; we can start with a single adapter and later extend to group-wise fusion. |
| Loss mixing logic (`loss_alpha/beta/theta/gamma`) | same file | Encodes the dual-objective scheduling; we can port the weighting scheme into our trainer to balance reasoning vs. classification tasks. |
| Data tools for reasoning reflection (`dataset/tools/getreflection.py`, etc.) | `previousWorks/ThinkRec/dataset/tools` | Scripts for harvesting mispredicted cases and generating rationales that can seed our `reason` field. |
| Config structure (`train_configs/...`) | repo root | Guides how to expose YAML-driven hyperparameters (loss weights, prompt templates, LoRA ranks). |

## 3. Path to Upgrade Task 1 Training Script

Short term adjustments to `agentic_recommender/finetune/qlora_finetune.py`:
- Extend the dataset loader to expect structured fields (`User_emb`, `TargetItem_emb`, text prompt components) and build prompt strings mirroring ThinkRec templates.
- Introduce dual-mode batches: mix reasoning and classification samples, attach metadata to decide which loss to apply.
- Swap the vanilla `SFTTrainer` for a custom `Trainer` subclass that computes the ThinkRec-style `loss = α·L_rec + β·L_reason` based on sample flags. This brings us closer to `forward_v3` behaviour without duplicating the entire codebase.

Medium term, consider factoring ThinkRec modules into reusable packages:
1. **Shared embedding projector**: extract `llama_proj` initialization + forward pass into `agentic_recommender/finetune/projectors.py` so both ThinkRec-derived models and our trainer can reuse consistent mapping from collaborative vectors to token embeddings.
2. **Prompt composer**: port `prompt_based_encode_v3` and related placeholder replacement into `agentic_recommender/finetune/prompting.py`, parameterized by prompt templates defined in our config files.
3. **Dual-objective trainer**: build a trainer class inspired by `MiniGPT4Rec_v3.forward_{v1,v2,v3}` that dynamically switches between the two losses. We can either:
   - (a) extend the Task 1 script to accept sample-level `mode` flags and compute the correct loss internally; or
   - (b) wrap ThinkRec’s model class, instantiating it with our data modules while pruning features we do not need (e.g., multi-adapter gating) for a simpler first iteration.

Longer term enhancements:
- Reuse ThinkRec’s clustering-based adapter routing when we have sufficient user diversity. The adapters can be trained via the same pipeline; we just need to generate `user2group.csv` from our embeddings and plug it into the ported `set_user2group` utility.
- Integrate the reflection data generation loop so that mispredicted samples automatically populate reasoning prompts, mirroring ThinkRec’s iterative improvement cycle.

In summary, we should start by adapting our basic QLoRA finetuner to emit ThinkRec-style prompts and losses, while progressively upstreaming projection and prompt utilities from the original repo to avoid re-implementing complex embedding-handling code.
