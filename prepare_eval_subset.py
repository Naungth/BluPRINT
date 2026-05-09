import json
import shutil
from pathlib import Path

def prepare_subset(jsonl_path: str, dataset_dir: str, hint_dir: str, output_base: str):
    jsonl_file = Path(jsonl_path)
    base_dir = Path(output_base)
    data_path = Path(dataset_dir)
    hint_path = Path(hint_dir)
    
    # Create the new clean directories
    real_out = base_dir / "eval_real"
    hint_out = base_dir / "eval_hints"
    real_out.mkdir(parents=True, exist_ok=True)
    hint_out.mkdir(parents=True, exist_ok=True)

    with open(jsonl_file, "r") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            # Extract JUST the filename (ignores mismatching folder names like 'source_images')
            source_name = Path(record["source"]).name
            target_name = Path(record["target"]).name
            
            # Recursively search for the actual files in your real directories
            found_target = list(data_path.rglob(target_name))
            found_hint = list(hint_path.rglob(source_name))

            if found_target and found_hint:
                shutil.copy2(found_hint[0], hint_out / source_name)
                shutil.copy2(found_target[0], real_out / target_name)
            else:
                if not found_target:
                    print(f"Warning: Could not find real image {target_name} in {dataset_dir}")
                if not found_hint:
                    print(f"Warning: Could not find hint image {source_name} in {hint_dir}")

    print(f"\nDone! Created isolated folders with exactly {sum(1 for _ in real_out.iterdir())} images:")
    print(f"Real images: {real_out.resolve()}")
    print(f"Hint images: {hint_out.resolve()}")

if __name__ == "__main__":
    # We point the script directly to your actual working directories
    prepare_subset("test_10.jsonl", "./dataset", "./hint_images", "./eval_subset")