from share import *

import torch
import pytorch_lightning as pl
import json
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

import torch.autograd.graph
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

# Configs
resume_path = './models/control_sd15_canny.pth'
batch_size = 64
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

checkpoint_callback = ModelCheckpoint(
    dirpath="./models/",
    filename="control_sd15_canny_finetuned-{epoch:02d}-{step}",
    save_top_k=2,
    save_last=True,
    monitor="val_loss",
    mode="min",
)

trainer = pl.Trainer(accelerator='gpu', devices=4, strategy='ddp_find_unused_parameters_true', precision="16-mixed", callbacks=[logger, TQDMProgressBar(), checkpoint_callback], max_epochs=50)

full_dataset = MyDataset()

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=False, persistent_workers=True)

val_indices = val_dataset.indices
val_data_items = [full_dataset.data[idx] for idx in val_indices]

# Save to jsonl file
output_path = '/home/ubuntu/BluPRINT/val_dataset.jsonl'
with open(output_path, 'w') as f:
    for item in val_data_items:
        f.write(json.dumps(item) + '\n')

print(f"Saved {len(val_data_items)} validation samples to {output_path}")

torch.set_float32_matmul_precision('medium')
# Train!
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)