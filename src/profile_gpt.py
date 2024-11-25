import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def profile_gpt2_training(
    model_name="gpt2",
    batch_sizes=[1, 2, 4],
    seq_lengths=[16, 32, 64],
    num_training_steps=10
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running training profiling on {device}...")

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    profiling_results = []

    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            print(f"\nBatch Size: {batch_size}, Sequence Length: {seq_length}")

            input_text = "The future of AI is bright." * (seq_length // 5)
            inputs = tokenizer([input_text] * batch_size, return_tensors="pt", padding=True, truncation=True).to(device)
            labels = inputs["input_ids"]

            torch.cuda.synchronize()
            start_time = time.time()
            total_training_time = 0

            for _ in range(num_training_steps):
                optimizer.zero_grad()

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

                loss.backward()

                optimizer.step()

                torch.cuda.synchronize() 
                total_training_time += time.time() - start_time

            gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2) 
            
            avg_training_time = total_training_time / num_training_steps
            print(f"Avg Training Time/Step: {avg_training_time:.4f}s, GPU Memory: {gpu_memory:.2f}MB")

            profiling_results.append({
                "batch_size": batch_size,
                "seq_length": seq_length,
                "avg_training_time": avg_training_time,
                "gpu_memory": gpu_memory
            })

    return profiling_results

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seq_lengths", nargs="+", type=int, default=[16, 32, 64])
    args = parser.parse_args()

    results = profile_gpt2_training(batch_sizes=args.batch_sizes, seq_lengths=args.seq_lengths)

    print("\nTraining Profiling Complete! Results:")
    for result in results:
        print(result)