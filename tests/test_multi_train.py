# import os
# # Turn off XLA JIT
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit=false"
# os.environ["XLA_FLAGS"]    = "--xla_cpu_disable_jit"

# import tensorflow as tf
# # Just in case: also disable via the Python API
# tf.config.optimizer.set_jit(False)


from textattack.attack_recipes import TextBuggerLi2018, PWWSRen2019, DeepWordBugGao2018
from textattack.attack import Attack
from textattack.shared import AttackedText
from textattack import Trainer, TrainingArgs
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

import os
import time
import wandb
from wandb.tensorboard import patch

import copy
import torch.nn.functional as F
import unicodedata

os.environ["SSL_CERT_FILE"] = os.path.join(os.path.dirname(wandb.__file__), "data", "cacert.pem")
patch(tensorboard=True)

# Initialize a W&B run and record your hyperparameters
wandb.init(
    project="adv-train-agnews",           # your W&B project name
    name=f"advrun-{int(time.time())}",    # unique run name
    config={                              # mirror your TrainingArgs here
        "model_name": "bert-base-uncased-ag-news",
        "num_clean_epochs": 1,
        "num_epochs": 3,
        "attack_epoch_interval": 1,
        "num_train_adv_examples": 0.2,
        "query_budget_train": 5000,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
    },
    mode="online"                         # or "offline" if you need to sync later
)

# EMA helper
def ema_update(teacher, student, tau=0.999):
    with torch.no_grad:
        for pt, ps in zip(teacher.parameters(), student.parameters()):
            pt.data.mul_(tau).add_(ps.data, alpha= 1 - tau)
        
        for bt, bs in zip(teacher.buffers(), student.buffers()):
            bt.data.mul_(tau).add_(bs.data, alpha= 1 - tau)
            
            
# VAE helper
@torch.no_grad()
def vae_noiser(texts, tokenizer, vae, mode="jitter", sigma = 0.2, max_edit = 2):
    """
    texts -> [str]; returns tilde_texts -> [str]
    Constraint guards keep changes tiny & readable.
    """
    
    enc = tokenizer(texts, return_tensor="pt", padding=True, truncation=True).to(next(vae.parameter()).device)
    mu, logvar = vae.encode(enc["input_ids"])
    z = mu if mode == "recon" else mu + sigma * torch.randn_like(mu)
    
    out_ids = vae.greedy_decode(
        z,
        start_id=(tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id or 0),
        max_len=enc["input_ids"].size(1),
        temperature=0.8,
    )
    xs = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    # light guards (implement your own levenshtein if you have one)
    def looks_bad(s):
        return any(ord(ch) < 32 and ch not in "\t\n\r" for ch in s)
    def safe_variant(s0, s):
        s = unicodedata.normalize("NFC", s)
        # optional: plug in your levenshtein(s, s0) and reject if > max_edit
        return s0 if looks_bad(s) else s

    return [safe_variant(s0, s) for s0, s in zip(texts, xs)]

def kl_softmax(p_logits, q_logits, T=2.0):
    p = torch.log_softmax(p_logits / T, dim=-1)
    q = torch.log_softmax(q_logits / T, dim=-1)
    
    return (torch.exp(p) * (p - q).sum(dim=-1).mean())


# get the result
def ta_adv_batch(attack: Attack, texts, labels):
    """
    Returns one adversarial text per input text.
    Works with your RandomMixtureAttack.
    """
    # small in-memory "dataset"
    data = list(zip(texts, labels))
    adv_texts = []
    for (x, y), res in zip(data, attack.attack_dataset(data)):
        # TA APIs vary slightly by version; try new, then old:
        s = None
        try:
            if getattr(res, "perturbed_result", None):
                s = res.perturbed_result.attacked_text.text
        except Exception:
            pass
        if s is None:
            try:
                s = res.perturbed_text()
            except Exception:
                s = None
        adv_texts.append(s if s else x)
    return adv_texts

def train_with_vae_consistency(model, tokenizer, attack, train_ds, eval_ds, device):
    # --- EMA teacher ---
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- optimizer (match your TrainingArgs) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # --- batching utility over TextAttack's HuggingFaceDataset ---
    def batches(dataset, batch_size=8):
        # the TA dataset supports __len__/__getitem__
        n = len(dataset)
        idx = list(range(n))
        random.shuffle(idx)
        for i in range(0, n, batch_size):
            sl = idx[i:i+batch_size]
            texts = [dataset[j][0] for j in sl]
            labels = [dataset[j][1] for j in sl]
            yield texts, torch.tensor(labels, dtype=torch.long)

    # --- (optional) load your pretrained VAE here ---
    # vae = TinyTextVAE.from_pretrained("path/to/vae").to(device).eval()
    vae = ...  # <-- load your VAE
    gamma = 0.3          # consistency weight
    Tkl   = 2.0          # temperature
    tau   = 0.999        # EMA decay
    sigma_sched = lambda e: min(0.3, 0.1 + 0.05*e)

    num_clean_epochs = 1
    num_epochs = 3

    global_step = 0
    for epoch in range(1, num_clean_epochs + num_epochs + 1):
        model.train()
        use_attack = (epoch > num_clean_epochs)  # adversarial from your TA attacks
        epoch_loss, epoch_clean, epoch_adv, epoch_cons = 0.0, 0.0, 0.0, 0.0

        for texts, y in batches(train_ds, batch_size=8):
            # 1) Build adv texts using your existing mixed attack (after clean warmup)
            if use_attack:
                x_adv = ta_adv_batch(attack, texts, y.tolist())
            else:
                x_adv = texts

            # 2) Build VAE-perturbed neighbors for consistency
            x_tilde = vae_noiser(texts, tokenizer, vae, mode="jitter", sigma=sigma_sched(epoch), max_edit=2)

            # 3) Tokenize
            batch_clean = tokenizer(texts,   return_tensors="pt", padding=True, truncation=True).to(device)
            batch_adv   = tokenizer(x_adv,   return_tensors="pt", padding=True, truncation=True).to(device)
            batch_tilde = tokenizer(x_tilde, return_tensors="pt", padding=True, truncation=True).to(device)
            y = y.to(device)

            # 4) Forward
            logits_clean = model(**batch_clean).logits
            logits_adv   = model(**batch_adv).logits
            with torch.no_grad():
                logits_teacher = teacher(**batch_clean).logits
            logits_tilde = model(**batch_tilde).logits

            # 5) Losses
            loss_clean = F.cross_entropy(logits_clean, y)
            loss_adv   = F.cross_entropy(logits_adv,   y)   # swap to TRADES-KL if you prefer
            loss_cons  = kl_softmax(logits_teacher, logits_tilde, T=Tkl)

            loss = loss_clean + loss_adv + gamma * loss_cons

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema_update(teacher, model, tau=tau)

            # 6) Logging
            epoch_loss += loss.item()
            epoch_clean += loss_clean.item()
            epoch_adv   += loss_adv.item()
            epoch_cons  += loss_cons.item()
            global_step += 1

            wandb.log({
                "loss/total": loss.item(),
                "loss/clean": loss_clean.item(),
                "loss/adv": loss_adv.item(),
                "loss/cons": loss_cons.item(),
                "epoch": epoch,
                "step": global_step
            })

        print(f"[epoch {epoch}] "
              f"loss={epoch_loss:.3f} clean={epoch_clean:.3f} adv={epoch_adv:.3f} cons={epoch_cons:.3f}")


# create an atta`ck wrapper that mix different attacksgl
class RandomMixtureAttack(Attack):
    def __init__(self, attacks, query_budget):
        self.attacks = attacks
        examplar = attacks[0] # pick any one of them to initialize base attributes
        
        for atk in self.attacks:
            atk.goal_function.num_queries  = 0
            atk.goal_function.query_budget = query_budget
            
            atk.constraints = [
                c for c in atk.constraints
                if "semantics" not in c.__class__.__module__
            ]
            
        
        super().__init__(
            goal_function=examplar.goal_function,
            constraints=examplar.constraints,
            transformation=examplar.transformation,
            search_method=examplar.search_method
        )
        
    def attack(self, initial_result, ground_truth_output):
        # choose a random attack
        chosen = random.choice(self.attacks)
        
        # reset per-example counters & budget
        if hasattr(chosen.goal_function, "num_queries"):
            chosen.goal_function.num_queries = 0
        
            
        result = chosen.attack(initial_result, ground_truth_output)
        # robust logging
        status = getattr(result, "goal_status", "Skipped")
        success = int(status == "Successful")
        queries = getattr(chosen.goal_function, "num_queries", None)

        wandb.log({
            "attack_chosen": type(chosen).__name__,
            "attack_status": status,          # e.g., "Successful", "Failed", or "Skipped"
            "attack_success": success,        # 1/0
            "attack_skipped": int(status=="Skipped"),
            **({"queries_used": int(queries)} if isinstance(queries, (int, float)) else {})
        })
        
        return result

def main():
    # setup parameter for checkpointing
    ckpt_root = "adv_train_outputs"
    last_dir = os.path.join(ckpt_root, "last_model")
    
    # setup the model wrapper
    model_name = "textattack/bert-base-uncased-ag-news"
    if os.path.isdir(last_dir):
        print(f"Resuming from {last_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(last_dir)
    else:   
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # list of attacks used in training
    attacks = [TextBuggerLi2018.build(model_wrapper),
            PWWSRen2019.build(model_wrapper),
            DeepWordBugGao2018.build(model_wrapper)]

    # prepare the dataset
    train_ds = HuggingFaceDataset("ag_news", split="train", dataset_columns=(["text"], "label"))
    eval_ds = HuggingFaceDataset("ag_news", split="test", dataset_columns=(["text"], "label"))

    # create the trainer
    training_args = TrainingArgs(
        num_clean_epochs=1,
        num_epochs=3,
        attack_epoch_interval=1,
        num_train_adv_examples=0.2,
        query_budget_train=64,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        output_dir="adv_train_outputs",
        log_to_tb=True,
        checkpoint_interval_steps=500,
        save_last=True,
    )
    
     # build your mixed attack as you already do
    mix_attack = RandomMixtureAttack(attacks, training_args.query_budget_train)

    # call custom loop instead of TextAttack Trainer
    train_with_vae_consistency(model, tokenizer, mix_attack, train_ds, eval_ds, device)
    
    
if __name__ == "__main__":
    print(">>> test_multi_train.py: Script is starting")
    main()
    print(">>> test_multi_train.py: Script has finished")