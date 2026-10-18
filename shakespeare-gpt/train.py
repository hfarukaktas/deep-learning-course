import os
import time
import torch
from model import GPT
from dataset import load_data, get_batch

EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6
BLOCK_SIZE = 256
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
WARMUP_ITERS = 100

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"

def get_learning_rate(iteration):
    if iteration < WARMUP_ITERS:
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    else:
        return LEARNING_RATE
@torch.no_grad()
def evaluate(model, train_data, test_data):
    model.eval()
    results = {}

    for name, data in [("train", train_data), ("test",test_data)]:
        total_loss = 0.0
        for _ in range(100):
            x,y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            x,y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x,y)
            total_loss += loss.item()
        results[name] = total_loss/100
    model.train()
    return results

@torch.no_grad()
def generate_sample(model, tokeizer):
    model.eval()

    prompt = "ROMEO: "
    prompt_ids = tokeizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device= DEVICE).unsqueeze(0)

    output_ids = model.generate(input_ids, max_new_tokens=200,temperature=0.8)

    model.train()
    return tokeizer.decode(output_ids[0])

if __name__ == "__main__":
    gpt = GPT(vocab_size=65, embedding_dim=384).to(DEVICE)
    train_data, test_data, tokenizer = load_data()
    print(generate_sample(gpt, tokenizer))
