from tutorial_dataset_jsonl import MyDataset


def main() -> None:
    train_jsonl = "../splits/train_90.jsonl"
    test_jsonl = "../splits/test_10.jsonl"

    with open(train_jsonl, "rt", encoding="utf-8") as f:
        train_count = sum(1 for _ in f if _.strip())
    with open(test_jsonl, "rt", encoding="utf-8") as f:
        test_count = sum(1 for _ in f if _.strip())

    print("Train split file:", train_jsonl, "records:", train_count)
    print("Test split file: ", test_jsonl, "records:", test_count)

    dataset = MyDataset(
        jsonl_path=train_jsonl,
        project_root="..",
        image_size=512,
    )

    print("Dataset size:", len(dataset))
    item = dataset[0]

    print("Prompt:", item["txt"])
    print(
        "jpg shape/range:",
        item["jpg"].shape,
        f"[{item['jpg'].min():.4f}, {item['jpg'].max():.4f}]",
    )
    print(
        "hint shape/range:",
        item["hint"].shape,
        f"[{item['hint'].min():.4f}, {item['hint'].max():.4f}]",
    )


if __name__ == "__main__":
    main()
