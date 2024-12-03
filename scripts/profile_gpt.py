import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
from tqdm import tqdm
import os
import json
import time
import psutil
import GPUtil

def get_device_properties(device=0):
    props = torch.cuda.get_device_properties(device)
    return {
        "name": torch.cuda.get_device_name(device),
        "totalMemory": props.total_memory,
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count
    }

def log_resource_usage():
    try:
        gpu = GPUtil.getGPUs()[0]  
        return {
            "gpu_util": gpu.memoryUtil * 100, 
            "gpu_memory": gpu.memoryUsed,
            "cpu_util": psutil.cpu_percent(), 
            "memory_util": psutil.virtual_memory().percent
        }
    except Exception as e:
        print(f"Error fetching resource usage: {e}")
        return {
            "gpu_util": 0,
            "gpu_memory": 0,
            "cpu_util": 0,
            "memory_util": 0
        }


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):
        def __init__(self, tokenizer, num_samples=10000):  # Dataset size for realistic load
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.samples = [
                "This is a sample text for training GPT2. It needs to be longer to better simulate real training conditions.",
                "Another sample text with more variety. Adding longer sentences for more realistic training behavior.",
                "GPT2 is being fine-tuned on this dataset. This sentence contains different words to improve generalization.",
                "A longer paragraph helps simulate real-world usage. This is essential for training with meaningful workloads.",
                "Random text for testing the GPT2 model. Texts like these ensure more diverse training data.",
            ]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            input_text = self.samples[idx % len(self.samples)]
            encoded = self.tokenizer(
                input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length"
            )
            return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    dataset = DummyDataset(tokenizer, num_samples=10000)  # Increased dataset size
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    return model, dataloader, optimizer

def profile_training():
    model, dataloader, optimizer = train_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = "../data/logs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    
    metrics_data = []
    trace_data = {
        "traceEvents": [],
        "deviceProperties": [get_device_properties()]
    }

    num_epochs = 10
    warmup_epochs = 2

    def trace_handler(p):
        output = p.key_averages().table(sort_by="cpu_time_total", row_limit=-1)
        metrics = p.key_averages()
        events = []
        
        for evt in metrics:
            event_data = {
                'name': evt.key,
                'ph': 'X',
                'cat': 'kernel' if evt.device_time_total > 0 else 'cpu_op',
                'pid': 0,
                'tid': 0,
                'ts': evt.cpu_time_total, 
                'dur': evt.device_time_total if evt.device_time_total > 0 else evt.cpu_time_total,
                'args': {
                    'gpu_usage': evt.device_time_total, 
                    'cpu_usage': evt.cpu_time_total,   
                    'memory_usage': evt.device_memory_usage,
                    'duration': evt.cpu_time_total + evt.device_time_total, 
                }
            }
            events.append(event_data)
            
        trace_data["traceEvents"].extend(events)
        print(output)

        trace_file_path = f"{output_dir}/trace_latest.json"
        with open(trace_file_path, 'w') as f:
            json.dump(trace_data, f, indent=4)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=2),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            model.train()
            epoch_start = time.time()

            for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader)):
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

                step_metrics = log_resource_usage()
                with record_function("forward"):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss

                with record_function("backward"):
                    optimizer.zero_grad()
                    loss.backward()

                with record_function("optimizer"):
                    optimizer.step()

                prof.step()

                step_metrics.update({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "loss": loss.item(),
                })
                metrics_data.append(step_metrics)

                if batch_idx % 10 == 0:
                    pd.DataFrame(metrics_data).to_csv(f"{output_dir}/metrics_{timestamp}.csv", index=False)

            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s")

if __name__ == "__main__":
    profile_training()

