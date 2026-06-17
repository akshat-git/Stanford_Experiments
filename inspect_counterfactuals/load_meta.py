import json
import pandas as pd
from form_dataset import InspectDataset

# Load data
dataset = InspectDataset(split="train")

# Prepare sample data index
samples = []
for idx in range(len(dataset)):
    img, cond, lbl = dataset[idx]
    sample = {
        "image_id": f"PE{idx:06d}",
        "split": "train",  # Example: could be "train" or "test"
        "condition": cond.tolist(),
        "labels": lbl.tolist(),
    }
    samples.append(sample)

# Save index to JSON file
with open('sample_index.json', 'w') as f:
    json.dump(samples, f, indent=2)
