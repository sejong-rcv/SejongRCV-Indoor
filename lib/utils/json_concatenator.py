import json
import os
import numpy as np

def json_concatenator(json1, json2, save_path):
    
    json_all = []
    with open(os.path.join(json1), 'r') as f:
        json1_data = json.load(f)
    
    with open(os.path.join(json2), 'r') as f:
        json2_data = json.load(f)

    json_all.extend(json1_data)
    json_all.extend(json2_data)

    with open(os.path.join(save_path), 'w') as f:
        json.dump(json_all, f, indent=1)
    
    print("Go to submit!")

if __name__ == "__main__":
    json_concatenator(json1, json2, save_path)