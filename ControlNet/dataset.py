import json
import os
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/ubuntu/BluPRINT/training_data_qwen.jsonl', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = os.path.basename(item['target'])
        target_filename = os.path.basename(item['source'])
        prompt = item['prompt']

        # Use os.path.join to safely construct the file paths
        source = cv2.imread(os.path.join('/home/ubuntu/BluPRINT/dataset', source_filename))
        target = cv2.imread(os.path.join('/home/ubuntu/BluPRINT/hint_images', target_filename))

        # If the image wasn't found, raise a clear error rather than failing in cvtColor
        if source is None:
            raise FileNotFoundError(f"Source image not found: {os.path.join('/home/ubuntu/BluPRINT/dataset', source_filename)}")
        if target is None:
            raise FileNotFoundError(f"Target image not found: {os.path.join('/home/ubuntu/BluPRINT/hint_images', target_filename)}")

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize images to a consistent shape to prevent DataLoader batching errors
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

