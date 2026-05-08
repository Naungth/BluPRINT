import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from tutorial_dataset_jsonl import MyDataset


# Configs (aligned with docs/train.md defaults)
# Fine-tuning from pretrained Canny ControlNet; use Canny-style hints for best results.
resume_path = "./models/control_sd15_canny.pth"
batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# Dataset paths (directory-aware relative to ControlNet/)
# Use split_jsonl_train_test.py to create these before training.
jsonl_path = "../splits/train_90.jsonl"
project_root = ".."


def main() -> None:
    # First use CPU to load models. Lightning will move to GPU(s).
    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    dataset = MyDataset(
        jsonl_path=jsonl_path,
        project_root=project_root,
        image_size=512,
    )
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
    )
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
