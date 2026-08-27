import os
import json
import hashlib
import pandas as pd
import numpy as np

def generate_f0_splits():
    print("--- RUNNING F0.1 GOVERNED DATA SCAFFOLD ---")
    
    # 1. Define seeds (Mandatory F0 seeds: 1103 and 2207)
    SEEDS = [1103, 2207]
    
    # 2. Schema definition placeholder
    schema_info = {
        "dataset": "CIC-IDS2017",
        "seeds": SEEDS,
        "splits": ["train", "utility_val", "gate_dev", "sealed_test"],
        "governance": "Sealed test partition remains uninspected during Phase F0"
    }
    
    # Write Schema Hash
    schema_str = json.dumps(schema_info, sort_keys=True)
    schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
    schema_info["schema_hash"] = schema_hash
    
    os.makedirs("configs/data", exist_ok=True)
    with open("configs/data/cicids2017_schema.json", "w") as f:
        json.dump(schema_info, f, indent=4)
        
    print(f"Schema configuration generated | Hash: {schema_hash}")
    print("Data scaffold successfully initialized.")

if __name__ == "__main__":
    generate_f0_splits()
