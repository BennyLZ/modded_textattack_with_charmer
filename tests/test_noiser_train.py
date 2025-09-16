#!/usr/bin/env python
import os, time, random, copy, unicodedata
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)

import wandb
try:
    from wandb.integration.tensorboard import patch as tb_patch
except Exception:
    tb_patch = None

from textattack.attack_recipes import TextBuggerLi2018, PWWSRen2019, DeepWordBugGao2018, charmer_2024
from textattack.attack import Attack
from textattack import Attacker, AttackArgs
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset as TADataset

# -------------------
# Environment hygiene
# -------------------
# Make certs available on execute nodes
os.environ["SSL_CERT_FILE"] = os.path.join(os.path.dirname(wandb.__file__), "data", "cacert.pem")
# Keep huggingface caches local
os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".cache", "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
# Quiet TF spam if imported indirectly
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# If you shipped NLTK data, uncomment:
# os.environ["NLTK_DATA"] = os.path.join(os.getcwd(), "nltk_data")
# os.environ["TEXTATTACK_SKIP_NLTK_DOWNLOAD"] = "1"

if tb_patch:
    tb_patch(root_logdir="logs")  # only matters if you create TB SummaryWriters later

wandb.init(
    project="adv-train-agnews",
    name=f"mlm-consistency-{int(time.time())}",
    config={
        "model_name": "textattack/bert-base-uncased-ag-news",
        "num_clean_epochs": 1,
        "num_epochs": 2,
        "attack_interval": 3,
        "attack_query_budget": 64,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "consistency_weight": 0.3,
        "consistency_temp": 2.0,
        "mlm_mask_prob": 0.10,
        "mlm_max_masks": 2,
    },
    mode=os.environ.get("WANDB_MODE", "online"),  # set WANDB_MODE=offline on CHTC
)

# -------------------
# Helpers
# -------------------
def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, tau: float = 0.999, update_buffers: bool = True):
    """EMA update teacher <- tau*teacher + (1-tau)*student.
    Float buffers are EMA'd; non-float buffers are copied.
    """
    with torch.no_grad():
        # parameters
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.mul_(tau).add_(p_s, alpha=1 - tau)

        if update_buffers:
            t_bufs = dict(teacher.named_buffers())
            s_bufs = dict(student.named_buffers())
            for name, b_t in t_bufs.items():
                b_s = s_bufs[name]
                if b_t.dtype.is_floating_point:
                    b_t.mul_(tau).add_(b_s, alpha=1 - tau)
                else:
                    # integer / bool buffers: just mirror student
                    b_t.copy_(b_s)

def kl_softmax(p_logits: torch.Tensor, q_logits: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """KL(softmax(p/T) || softmax(q/T)) averaged over batch."""
    p = torch.log_softmax(p_logits / T, dim=-1)
    q = torch.log_softmax(q_logits / T, dim=-1)
    return (torch.exp(p) * (p - q)).sum(dim=-1).mean()

# @torch.no_grad()
# def mlm_noiser(
#     texts,
#     tokenizer: AutoTokenizer,
#     mlm: AutoModelForMaskedLM,
#     mask_prob: float = 0.10,
#     max_masks: int = 2,
#     topk: int = 1,
# ):
#     """Make tiny, plausible edits via a masked-LM. Returns List[str] same length."""
#     # Normalize to List[str]
#     if isinstance(texts, dict) and "text" in texts:
#         texts = texts["text"]
#     elif isinstance(texts, (list, tuple)) and texts and isinstance(texts[0], dict) and "text" in texts[0]:
#         texts = [ex["text"] for ex in texts]
#     elif isinstance(texts, (list, tuple)) and texts and isinstance(texts[0], (list, tuple)):
#         texts = [t[0] for t in texts]
#     texts = [str(t) for t in texts]

#     device = next(mlm.parameters()).device
#     batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
#     input_ids = batch["input_ids"].clone()
#     attn = batch["attention_mask"]
#     mask_id = tokenizer.mask_token_id

#     if mask_id is None:
#         # No [MASK] token (e.g., GPT2 tokenizer) — just return originals
#         return texts

#     specials = set(tokenizer.all_special_ids or [])
#     B, L = input_ids.shape
#     for i in range(B):
#         # Avoid masking special/pad IDs and try not to hit pure digits/URLs
#         positions = []
#         for j in range(L):
#             if attn[i, j].item() != 1:
#                 continue
#             tok = input_ids[i, j].item()
#             if tok in specials:
#                 continue
#             s = tokenizer.convert_ids_to_tokens(tok)
#             if s and ("http" in s or s.isnumeric()):
#                 continue
#             positions.append(j)

#         m = min(max_masks, max(1, int(len(positions) * mask_prob))) if positions else 0
#         if m > 0:
#             sel = random.sample(positions, m)
#             input_ids[i, sel] = mask_id

#     logits = mlm(input_ids=input_ids, attention_mask=attn).logits  # [B,L,V]
#     if topk <= 1:
#         preds = logits.argmax(-1)
#     else:
#         preds = torch.topk(logits, topk, dim=-1).indices[..., 0]

#     replaced = batch["input_ids"].clone()
#     mask_pos = (input_ids == mask_id)
#     replaced[mask_pos] = preds[mask_pos]

#     xs = tokenizer.batch_decode(replaced, skip_special_tokens=True)

#     def looks_bad(s):
#         return any(ord(ch) < 32 and ch not in "\t\n\r" for ch in s)
#     out = []
#     for s0, s in zip(texts, xs):
#         s = unicodedata.normalize("NFC", s)
#         out.append(s0 if looks_bad(s) else s)
#     return out


@torch.no_grad()
def mlm_noiser_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    mlm: AutoModelForMaskedLM,
    mask_prob: float = 0.10,
    max_masks: int = 2,
    topk: int = 1,
):
    """
    Like your mlm_noiser, but works on token IDs directly.
    Returns (new_input_ids) with same shape; shares attention_mask.
    """
    device = input_ids.device
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        return input_ids  # no [MASK] available

    specials = set(tokenizer.all_special_ids or [])
    B, L = input_ids.shape
    masked_ids = input_ids.clone()

    # choose mask positions per example
    for i in range(B):
        pos = []
        for j in range(L):
            if attention_mask[i, j].item() != 1:   continue
            tok = input_ids[i, j].item()
            if tok in specials:                    continue
            tokstr = tokenizer.convert_ids_to_tokens(tok)
            if tokstr and ("http" in tokstr or tokstr.isnumeric()):  # skip urls/digits
                continue
            pos.append(j)
        m = min(max_masks, max(1, int(len(pos) * mask_prob))) if pos else 0
        if m > 0:
            for j in random.sample(pos, m):
                masked_ids[i, j] = mask_id

    logits = mlm(input_ids=masked_ids, attention_mask=attention_mask).logits  # [B,L,V]
    if topk <= 1:
        preds = logits.argmax(-1)
    else:
        preds = torch.topk(logits, topk, dim=-1).indices[..., 0]

    out_ids = input_ids.clone()
    mask_pos = (masked_ids == mask_id)
    out_ids[mask_pos] = preds[mask_pos]
    return out_ids


def ta_adv_batch(attack, texts, labels):
    texts = ensure_str_list(texts)                 # normalize to List[str]
    data = list(zip(texts, labels))                # List[Tuple[str, int]]
    ta_ds = TADataset(data)                        # <-- wrap in TA Dataset
    args = AttackArgs(
        num_examples=len(data),
        shuffle=False,
        disable_stdout=True,
        parallel=False,
        # query_budget=...   # optional; you already set on goal_function
    )
    attacker = Attacker(attack, ta_ds, args)       # <-- pass Dataset, not list
    adv_texts = []
    for (x, _), res in zip(data, attacker.attack_dataset()):
        s = None
        if getattr(res, "perturbed_result", None):
            s = res.perturbed_result.attacked_text.text
        elif hasattr(res, "perturbed_text"):
            try: s = res.perturbed_text()
            except Exception: s = None
        adv_texts.append(s if s else x)
    return adv_texts

def ensure_str_list(seq):
    out = []
    for t in seq:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, (list, tuple)):
            out.append(" ".join(map(str, t)))   # join multi-field inputs
        elif isinstance(t, dict) and "text" in t:
            out.append(str(t["text"]))
        else:
            out.append(str(t))
    return out

def batches(dataset, batch_size=8):
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    for i in range(0, n, batch_size):
        sl = idx[i:i+batch_size]
        texts = [dataset[j][0] for j in sl]
        labels = [dataset[j][1] for j in sl]
        texts = ensure_str_list(texts)              # <-- normalize here
        yield texts, torch.tensor(labels, dtype=torch.long)

# -------------------
# Mixed attack wrapper
# -------------------
class RandomMixtureAttack(Attack):
    """Pick one of several built attacks per example; preserve constraints/search per base attack."""
    def __init__(self, attacks, query_budget: int):
        self.attacks = attacks
        examplar = attacks[0]

        for atk in self.attacks:
            if hasattr(atk.goal_function, "num_queries"):
                atk.goal_function.num_queries = 0
            if hasattr(atk.goal_function, "query_budget"):
                atk.goal_function.query_budget = query_budget
            # Optional: drop heavy semantic constraints for speed during training
            atk.constraints = [c for c in atk.constraints if "semantics" not in c.__class__.__module__]

        super().__init__(
            goal_function=examplar.goal_function,
            constraints=examplar.constraints,
            transformation=examplar.transformation,
            search_method=examplar.search_method,
        )

    def attack(self, initial_result, ground_truth_output):
        chosen = random.choice(self.attacks)
        if hasattr(chosen.goal_function, "num_queries"):
            chosen.goal_function.num_queries = 0

        result = chosen.attack(initial_result, ground_truth_output)

        # Light logging
        status = getattr(result, "goal_status", "Skipped")
        success = int(status == "Successful")
        queries = getattr(chosen.goal_function, "num_queries", None)
        log = {
            "attack/chosen": type(chosen).__name__,
            "attack/status": status,
            "attack/success": success,
        }
        if isinstance(queries, (int, float)):
            log["attack/queries"] = int(queries)
        wandb.log(log)

        return result

# -------------------
# Training Loop
# -------------------
def train_with_mlm_consistency(model, tokenizer, attack, train_ds, eval_ds, device):
    cfg = wandb.config
    model.train()
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # MLM noiser model (use same base as classifier tokenizer)
    # You can also use "textattack/bert-base-uncased-ag-news" tokenizer with this MLM.
    mlm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device).eval()

    def batches(dataset, batch_size=8):
        n = len(dataset)
        idx = list(range(n))
        random.shuffle(idx)
        for i in range(0, n, batch_size):
            sl = idx[i:i+batch_size]
            texts = [dataset[j][0] for j in sl]
            labels = [dataset[j][1] for j in sl]
            yield texts, torch.tensor(labels, dtype=torch.long)

    num_clean_epochs = cfg.num_clean_epochs
    num_epochs = cfg.num_epochs
    gamma = cfg.consistency_weight
    Tkl = cfg.consistency_temp

    global_step = 0
    for epoch in range(1, num_clean_epochs + num_epochs + 1):
        use_attack = (epoch > num_clean_epochs) and ( (epoch - num_clean_epochs) % cfg.attack_interval == 0 )
        epoch_loss = epoch_clean = epoch_adv = epoch_cons = 0.0
        zero = torch.zeros([], device=device)

        for texts, y in batches(train_ds, batch_size=cfg.per_device_train_batch_size):
            # --- normalize inputs ---
            texts = ensure_str_list(texts)           # strings for TextAttack
            y_cpu  = y.long()                        # TA needs Python ints
            y      = y_cpu.to(device)                # model needs CUDA tensor

            # --- 1) Tokenize CLEAN exactly once ---
            batch_clean = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            batch_clean = {k: v.to(device) for k, v in batch_clean.items()}

            # --- 2) Build TILDE on token IDs (no decode/re-encode) ---
            input_ids_tilde = mlm_noiser_ids(
                input_ids=batch_clean["input_ids"],
                attention_mask=batch_clean["attention_mask"],
                tokenizer=tokenizer,
                mlm=mlm,
                mask_prob=cfg.mlm_mask_prob,
                max_masks=cfg.mlm_max_masks,
            )
            # same attention mask as clean
            attn_tilde = batch_clean["attention_mask"]

            # --- 3) Adversarial branch: only strings need tokenization ---
            if use_attack:
                adv_texts = ta_adv_batch(attack, texts, y_cpu.tolist())
                batch_adv = tokenizer(
                    adv_texts, return_tensors="pt", padding=True, truncation=True
                )
                batch_adv = {k: v.to(device) for k, v in batch_adv.items()}
            else:
                batch_adv = None

            # --- 4) Forward passes (AMP optional) ---
            with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):
                logits_clean = model(**batch_clean).logits
                logits_tilde = model(input_ids=input_ids_tilde, attention_mask=attn_tilde).logits

                if batch_adv is not None:
                    logits_adv = model(**batch_adv).logits

                # Teacher consistency (keep) OR stop-grad consistency (faster)
                with torch.no_grad():
                    logits_teacher = teacher(**batch_clean).logits
                # Alternative: loss_cons = kl_softmax(logits_clean.detach(), logits_tilde, T=Tkl)
                loss_cons  = kl_softmax(logits_teacher, logits_tilde, T=Tkl)

                loss_clean = F.cross_entropy(logits_clean, y)
                loss_adv   = F.cross_entropy(logits_adv, y) if batch_adv is not None else zero

                loss = loss_clean + loss_adv + gamma * loss_cons

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema_update(teacher, model, tau=0.999)

            # Logging
            epoch_loss += float(loss.item())
            epoch_clean += float(loss_clean.item())
            epoch_adv   += float(loss_adv.item()) if use_attack else 0.0
            epoch_cons  += float(loss_cons.item())
            global_step += 1

            wandb.log({
                "loss/total": float(loss.item()),
                "loss/clean": float(loss_clean.item()),
                "loss/adv": float(loss_adv.item()) if use_attack else 0.0,
                "loss/cons": float(loss_cons.item()),
                "epoch": epoch,
                "step": global_step
            })

        print(f"[epoch {epoch}] loss={epoch_loss:.3f} clean={epoch_clean:.3f} adv={epoch_adv:.3f} cons={epoch_cons:.3f}")

        # Save a resumable checkpoint each epoch
        save_dir = os.path.join("adv_train_outputs", "last_model")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

# -------------------
# Main
# -------------------
def main():
    ckpt_root = "adv_train_outputs"
    last_dir = os.path.join(ckpt_root, "last_model")
    model_name = "textattack/bert-base-uncased-ag-news"

    if os.path.isdir(last_dir):
        print(f"Resuming from {last_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(last_dir)
        tokenizer = AutoTokenizer.from_pretrained(last_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Build base attacks
    attacks = [
        TextBuggerLi2018.build(model_wrapper),
        PWWSRen2019.build(model_wrapper),
        DeepWordBugGao2018.build(model_wrapper),
        charmer_2024.Charmer2024.build(model_wrapper)
    ]
    # Mix wrapper with a query budget
    mix_attack = RandomMixtureAttack(attacks, query_budget=wandb.config.attack_query_budget)

    # Datasets
    train_ds = HuggingFaceDataset("ag_news", split="train",
                              dataset_columns=(["text"], "label"))
    eval_ds  = HuggingFaceDataset("ag_news", split="test",
                              dataset_columns=(["text"], "label"))

    train_with_mlm_consistency(model, tokenizer, mix_attack, train_ds, eval_ds, device)

if __name__ == "__main__":
    print(">>> script starting")
    main()
    print(">>> script finished")
