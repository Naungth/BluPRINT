import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/test JSONL split.")
    parser.add_argument(
        "--input-jsonl",
        default="../training_data_qwen.jsonl",
        help="Path to source JSONL (relative to ControlNet/ by default).",
    )
    parser.add_argument(
        "--train-jsonl",
        default="../splits/train_90.jsonl",
        help="Output path for train split JSONL.",
    )
    parser.add_argument(
        "--test-jsonl",
        default="../splits/test_10.jsonl",
        help="Output path for test split JSONL.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).resolve()
    train_path = Path(args.train_jsonl).resolve()
    test_path = Path(args.test_jsonl).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError(f"train-ratio must be between 0 and 1, got {args.train_ratio}")

    records = read_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in: {input_path}")

    random.seed(args.seed)
    random.shuffle(records)

    train_count = int(len(records) * args.train_ratio)
    train_records = records[:train_count]
    test_records = records[train_count:]

    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)

    print(f"Input: {input_path} ({len(records)} records)")
    print(f"Train: {train_path} ({len(train_records)} records)")
    print(f"Test:  {test_path} ({len(test_records)} records)")
    print(f"Seed:  {args.seed}")


if __name__ == "__main__":
    main()
