''' run a small batch of the attack to test its performance and correctness '''

# prepare dataset
# from textattack.datasets import HuggingFaceDataset
from textattack.datasets import Dataset
from datasets import load_dataset
import itertools

# prepare attack component
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from textattack.attack_recipes import charmer_2024
from textattack import Attacker, AttackArgs


def chunked(iterator, size):
    while True:
        batch = list(itertools.islice(iterator, size))
        if not batch:
            break
        yield batch

def ag_stream():
    for example in raw_stream:
        # example["text"] is a string; we wrap it in a list for TextAttack
        yield ([example["text"]], example["label"])
        
def main():
    raw_stream = load_dataset("ag_news", split="test", streaming=True)

    model_name = "textattack/bert-base-uncased-ag-news"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    attack = charmer_2024.Charmer2024.build(model_wrapper, max_k=10, candidate_pos=10)

    # Log results to disk instead of keeping in memory
    attack_args = AttackArgs(
        num_examples=100,            # process the entire stream
        log_to_csv="ag_news_attacks.csv",
        disable_stdout=True,
        checkpoint_interval=None,
    )

    # Process in batches of, say, 100 examples at a time
    for i, batch in enumerate(chunked(ag_stream(), size=100)):
        print(f"\n=== Attacking batch #{i+1} ({len(batch)} examples) ===")
        small_ds = Dataset(batch)      # small list-backed Dataset with len=100
        attacker = Attacker(attack, small_ds, attack_args)
        results = attacker.attack_dataset()
        
        # display the result
        for result in results:
            print(result)
    
    
if __name__ == "__main__":
    main()
    

