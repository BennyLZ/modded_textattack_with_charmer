import textattack
import transformers
import torch
import torch.nn.functional as F
import wandb
import os
import time

class WandbLoggingTrainer(textattack.Trainer):
    def _log_metrics(self, metrics_dict, step=None):
        # Log metrics to wandb
        wandb.log(metrics_dict, step=step)

    def training_step(self, model, tokenizer, batch):
        t0 = time.perf_counter()

        loss_dict = super().training_step(model, tokenizer, batch)

        step_time = time.perf_counter() - t0

        lr = self.optimizer.param_groups[0]["lr"] if hasattr(self, "optimizer") else None
        grad_sq = 0.0
        if hasattr(self, "model"):
            for p in self.model.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    grad_sq += float((g*g).sum())
        grad_norm = grad_sq ** 0.5

        wandb.log({
            "time/step_s": step_time,
            "opt/lr": lr,
            "opt/grad_norm": grad_norm,
            **{f"loss/{k}": float(v) for k, v in loss_dict.items() if k != "total"},
            "loss/total": float(loss_dict.get("total", loss_dict.get("loss", 0.0))),
        }, step=getattr(self, "global_step", None))

        return loss_dict
    
    @torch.no_grad()
    def evaluation_step(self, model, tokenizer, batch):
        t0 = time.perf_counter()

        preds, targets = super().evaluation_step(model, tokenizer, batch)


        batch_loss = None
        batch_acc  = None

        try:
            if targets is not None:
                # Put both on same device for ops
                device = preds.device
                targets = targets.to(device)

                if preds.ndim > 1 and preds.size(-1) > 1 and targets.dtype in (torch.long, torch.int64):
                    # classification with logits
                    batch_loss = F.cross_entropy(preds, targets).item()
                    batch_acc  = (preds.argmax(-1) == targets).float().mean().item()
                else:
                    # regression or binary-prob: be conservative
                    # if binary probs/logits and targets are {0,1}, you may swap in BCEWithLogits here
                    batch_loss = F.mse_loss(preds.squeeze(), targets.float()).item()
                    batch_acc  = None
        except Exception:
            # keep evaluation robust even if shapes/types differ
            pass

        # 3) Log per-batch eval signals
        step = getattr(self, "global_step", None)
        wandb.log({
            "global_step": step,
            "eval/batch_time_s": time.perf_counter() - t0,
            **({"eval/batch_loss": batch_loss} if batch_loss is not None else {}),
            **({"eval/batch_acc":  batch_acc } if batch_acc  is not None else {}),
        }, step=step)

        # 4) Keep running aggregates for epoch-level metrics
        if not hasattr(self, "_eval_aggr"):
            self._eval_aggr = {"loss_sum": 0.0, "n": 0, "correct": 0}
        if batch_loss is not None:
            # weight by batch size (targets length) if available
            bsz = int(targets.numel()) if targets is not None else 1
            self._eval_aggr["loss_sum"] += float(batch_loss) * bsz
            self._eval_aggr["n"]        += bsz
        if batch_acc is not None and targets is not None:
            # recompute corrects from preds/targets for exact counting
            self._eval_aggr["correct"] += int((preds.argmax(-1).cpu() == targets.cpu()).sum())

        return preds, targets
    
    def evaluate(self, *args, **kwargs):
        # reset aggregates
        self._eval_aggr = {"loss_sum": 0.0, "n": 0, "correct": 0}
        out = super().evaluate(*args, **kwargs)

        # epoch-level means (clean set)
        n = max(1, self._eval_aggr["n"])
        mean_loss = self._eval_aggr["loss_sum"] / n
        acc = None
        if self._eval_aggr["correct"] > 0:
            acc = self._eval_aggr["correct"] / n

        step = getattr(self, "global_step", None)
        payload = {"global_step": step, "eval/loss": mean_loss}
        if acc is not None:
            payload["eval/acc"] = acc

        wandb.log(payload, step=step)
        return out

    # Additional method to run robust accuracy probe with wandb logging
    @torch.no_grad()
    def run_robust_probe(attack, dataset, n_samples=256, step=None):
        from textattack.attack_results import SuccessfulAttackResult, SkippedAttackResult
        total = succ = robust_correct = 0
        edits, pct_changed = [], []
        import time
        t0 = time.perf_counter()

        def tok(s): return s.split()
        for i, res in enumerate(attack.attack_dataset(dataset)):
            if i >= n_samples: break
            total += 1
            if isinstance(res, SuccessfulAttackResult):
                succ += 1
                clean = res.original_result.attacked_text.text
                adv   = res.perturbed_result.attacked_text.text
                w0, w1 = tok(clean), tok(adv)
                m = min(len(w0), len(w1))
                diff = sum(1 for j in range(m) if w0[j] != w1[j]) + abs(len(w0)-len(w1))
                edits.append(diff)
                pct_changed.append(diff / max(1, len(w0)))
            elif isinstance(res, SkippedAttackResult):
                # conservative: count as robust-correct only if original prediction was correct
                robust_correct += int(res.original_result.goal_status == "SUCCEEDED")
            else:
                # FailedAttackResult
                robust_correct += 1

        wandb.log({
            "global_step": step,
            "metrics/robust_acc": robust_correct / max(1, total),
            "attack/success_rate": succ / max(1, total),
            "attack/avg_edits": (sum(edits)/len(edits)) if edits else 0.0,
            "attack/avg_pct_changed": (sum(pct_changed)/len(pct_changed)) if pct_changed else 0.0,
            "attack/time_s": time.perf_counter() - t0,
            "attack/samples": total,
        }, step=step)
    

def main():
    # initialize wandb
    wandb.init(project="simple_adv_train", entity="liangbinz1599043", config={
        "model": "bert-base-uncased",
        "dataset": "imdb",
        "attack": "DeepWordBugGao2018",
        "num_epochs": 10,
        "num_clean_epochs": 4,
        "num_train_adv_examples": 1000,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "log_to_tb": True,
        },
        mode=os.environ.get("WANDB_MODE", "online"),
        sync_tensorboard=True,
        save_code=True
    )

    wandb.define_metric("global_step")                 # make it the x-axis
    wandb.define_metric("eval/*", step_metric="global_step")
    wandb.define_metric("attack/*", step_metric="global_step")
    wandb.define_metric("time/step_s", step_metric="global_step")
    wandb.define_metric("opt/lr",       step_metric="global_step")
    wandb.define_metric("opt/grad_norm",step_metric="global_step")
    wandb.define_metric("loss/*",       step_metric="global_step")

    model = transformers.AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    wandb.watch(model, log="all", log_freq=500)

    # We only use DeepWordBugGao2018 to demonstration purposes.
    attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
    train_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="train")
    eval_dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")

    # Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
    training_args = textattack.TrainingArgs(
        num_epochs=10,
        num_clean_epochs=4,
        num_train_adv_examples=1000,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        log_to_tb=True,
        log_to_wandb=True,
        wandb_project="simple_adv_train"
    )

    trainer = WandbLoggingTrainer(
        model_wrapper,
        "classification",
        attack,
        train_dataset,
        eval_dataset,
        training_args
    )
    
    trainer.run_robust_probe(attack, eval_dataset, n_samples=256, step=0)

    for epoch in range(int(training_args.num_epochs)):
        trainer.train(epoch)
        trainer.evaluate()
        trainer.run_robust_probe(attack, eval_dataset, n_samples=256, step=getattr(trainer, "global_step", None))

if __name__ == "__main__":
    main()