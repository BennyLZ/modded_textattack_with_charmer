import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import textattack
from textattack.attack_recipes import charmer_2024
from textattack import Trainer, TrainingArgs
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper

def main():
    # 1) prepare model + wrapper
    model_name = "textattack/bert-base-uncased-ag-news"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


    # 2) build charmer attack
    attack = charmer_2024.Charmer2024.build(model_wrapper, max_k=10, candidate_pos=10)

    # 3) load train and eval dataset
    train_ds = HuggingFaceDataset("ag_news", split="train", dataset_columns=(["text"], "label"))
    eval_ds = HuggingFaceDataset("ag_news", split="test", dataset_columns=(["text"], "label"))


    # 4) configure adversarial training setting
    training_args = TrainingArgs(
        num_clean_epochs=0,           # 1 epoch on clean data
        num_epochs=3,                 # total training epochs
        attack_epoch_interval=1,      # regenerate adversarial samples each epoch
        num_train_adv_examples=0.2,   # use 20% of train set per epoch
        query_budget_train=5000,      # max model queries per epoch
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        output_dir="adv_train_outputs",
        log_to_tb=True,                  
    )

    # 5) create and run trainer 
    trainer = Trainer(
        model_wrapper,
        task_type="classification",
        attack=attack,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_args=training_args,
    )
    trainer.train()


    # evaluate the model
    metric = trainer.evaluate()

if __name__ == "__main__":
    main()
