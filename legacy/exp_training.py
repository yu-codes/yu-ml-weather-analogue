# %%
# Standard library imports
import datetime
import gc
import json
import math
import os
import random
import sys

# Third-party library imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.dates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # 导入 LogNorm

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import FormatStrFormatter
from sklearn import preprocessing
from tqdm import tqdm

# PyTorch related imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchmetrics
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# PyTorch Lightning related imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, Logger

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import csv

# Wandb (Weights & Biases) related imports
import wandb

# %%
# current_dir = os.getcwd()
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))
from models.atmodist import OrdinalAtmodist, TripletAtmodist
from dataset.dataset import AtmodistDataset, OrdinalDataset, TripletDataset
from utils.utils import set_working_directory, load_json_config, ensure_directory_exists
from utils.utils_data import read_netcdf
from utils.utils_parsers import feature_map_parser, model_parser


try:
    config = load_json_config("config.json")
    # print(config)
except FileNotFoundError as e:
    print(e)
os.chdir(current_dir)
print(f"Working directory changed to: {os.getcwd()}")

seed = 42
pl.seed_everything(seed)  ## numpy, python, pytorch
if torch.cuda.is_available():
    print("cuda is available")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %% [markdown]
# ## Data Processing
# 

# %%
variable_list = ["d2m","u","v", "msl", "r"]
variables= ''.join(variable_list)
selected_frequency = 3
time_unit = "h"
resample_method = "none"
preprocessing_method = "standardized" 
year_range = (2001, 2020)
variables_path = f"../../../data/processed/{variables}_{selected_frequency}{time_unit}_{resample_method}_{preprocessing_method}_{year_range[0]}{year_range[1]}.h5"
atmosphere_data = read_netcdf(variables_path)

# %% [markdown]
# ## Model Building

# %%
class ShortcutDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, downscale=False):
        super(ShortcutDown, self).__init__()

        self.out_channels = out_channels
        self.stride = 2 if downscale else 1
        padding = (kernel_size - 1) // 2

        if in_channels != out_channels:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                bias=False,
            )
        else:
            self.conv = None 

    def forward(self, inputs):
        shortcut = (
            F.max_pool2d(inputs, kernel_size=1, stride=self.stride)
            if inputs.shape[1] == self.out_channels
            else self.conv(inputs)
        )
        return shortcut


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downscale=False):
        super().__init__()

        stride = 2 if downscale else 1
        padding = (kernel_size - 1) // 2

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = ShortcutDown(in_channels, out_channels, downscale=downscale)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, inputs):
        # First convolutional block
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)

        # Shortcut connection
        shortcut = self.shortcut(inputs)

        # Residual addition
        y = torch.add(shortcut, x)
        y = self.relu2(y)

        return y


class ResNet(nn.Module):

    def __init__(self, in_channels, out_channels_list, num_blocks=6):
        super(ResNet, self).__init__()
        self.resblocks = nn.ModuleList()

        for idx, out_channels in enumerate(out_channels_list):
            for block_idx in range(num_blocks):
                # downscale = ((idx == 0) or (idx == 2) or (idx == 3)) and (block_idx == 0)
                downscale = block_idx == 0
                resblock = ResBlockDown(in_channels, out_channels, downscale=downscale)
                self.resblocks.append(resblock)
                in_channels = out_channels

    def forward(self, inputs):
        x = inputs
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        channel_number_list,
        out_conv_channels=16,
        conv_kernel_size=8,
        maxpool_kernel_size=3,
        stride=2,
        # stride=1,
    ):
        super(Encoder, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=out_conv_channels,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            maxpool_kernel_size,
            stride=stride,
            padding=(maxpool_kernel_size - 1) // 2,
        )

        # Resblocks
        self.resnet = ResNet(out_conv_channels, channel_number_list)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.resnet(x)
        return x

# %%
class Atmodist(pl.LightningModule):

    def __init__(
        self,
        num_classes,
        res_in_channels,
        res_out_channels_list=(16, 32, 64),
        downstream_stack_channels=128,
        downstream_channels=64,
        downstream_kernel_size=3,
        downstream_fin_shape=1,
        lr=0.01,
        momentum=0.9,
    ):
        super(Atmodist, self).__init__()
        self.validation_step_outputs = []

        # Models
        self.num_classes = num_classes
        self.encoder = Encoder(res_in_channels, res_out_channels_list)
        self.conv1 = self.create_conv_block(
            downstream_stack_channels,
            downstream_channels,
            kernel_size=downstream_kernel_size,
            stride=2,
        )
        self.conv2 = self.create_conv_block(
            downstream_channels,
            downstream_channels,
            kernel_size=downstream_kernel_size,
            stride=1,
        )
        self.conv3 = self.create_conv_block(
            downstream_channels,
            downstream_channels,
            kernel_size=downstream_kernel_size,
            stride=1,
        )
        self.flat = nn.Flatten()
        self.linear = nn.Linear(
            # 128,
            downstream_channels * downstream_fin_shape * downstream_fin_shape,
            self.num_classes,
            bias=False,
        )
        # self.softmax = nn.Softmax(
        #     dim=-1
        # )  ## Remove the softmax layer because nn.CrossEntropyLoss already includes softmax
        # Loss function and hyperparameters
        self.drop_flat = nn.Dropout(0.2)
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum
        self.save_hyperparameters()

    def create_conv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=math.floor((kernel_size - 1) / 2),
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        img1, img2 = inputs
        atm1 = self.encoder(img1)
        atm2 = self.encoder(img2)
        x = torch.cat((atm1, atm2), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        # x = self.drop_flat(x)

        x = self.linear(x)
        # x = self.softmax(x)
        return atm1, atm2, x

    def forward_diff(self, inputs):
        img1, img2 = inputs
        atm1 = self.encoder(img1)  # 对第一个图像进行编码，得到特征表示 atm1
        atm2 = self.encoder(img2)  # 对第二个图像进行编码，得到特征表示 atm2

        # 计算差异张量
        diff = torch.abs(atm1 - atm2)  # 使用逐元素减法并取绝对值

        x = self.conv1(diff)
        x = self.conv2(x)

        x = self.flat(x)
        x = self.drop_flat(x)
        x = self.linear(x)
        x = self.drop_linear(x)
        return atm1, atm2, x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=8,
            cooldown=0,
            min_lr=1e-5,
            eps=4e-2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, batch, batch_idx):
        timestamp1, timestamp2, r1, r2, target = batch
        atm1, atm2, pred = self.forward((r1, r2))
        loss, accuracy = self.calculate_loss_and_accuracy(pred, target)
        self.log_metrics(loss, accuracy, prefix="train")

        return {"step": self.global_step, "loss": loss, "acc": accuracy}

    def validation_step(self, batch, batch_idx):
        timestamp1, timestamp2, r1, r2, target = batch
        atm1, atm2, pred = self.forward((r1, r2))
        loss, accuracy = self.calculate_loss_and_accuracy(pred, target)
        self.log_metrics(loss, accuracy, prefix="val")
        self.validation_step_outputs.append((pred, target))

        return {"val_loss": loss, "val_acc": accuracy}

    def on_validation_epoch_end(self):
        all_preds = []
        all_targets = []
        for pred, target in self.validation_step_outputs:
            all_preds.append(pred)
            all_targets.append(target)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        predicted_indices = torch.argmax(all_preds, dim=1).cpu().numpy()
        target_indices = torch.argmax(all_targets, dim=1).cpu().numpy()

        conf_matrix = confusion_matrix(target_indices, predicted_indices)
        self.log_confusion_matrix(conf_matrix)

        accuracy = accuracy_score(target_indices, predicted_indices)
        precision = precision_score(
            target_indices, predicted_indices, average="weighted", zero_division=0
        )
        recall = recall_score(
            target_indices, predicted_indices, average="weighted", zero_division=0
        )
        f1 = f1_score(target_indices, predicted_indices, average="weighted")
        class_report = classification_report(target_indices, predicted_indices)

        self.log("val_accuracy", accuracy)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        self.log("val_f1", f1)

        print(f"Classification Report:\n{class_report}")
        self.validation_step_outputs.clear()

    def calculate_loss_and_accuracy(self, pred, target):
        loss = self.loss(pred, target)
        predicted_indices = torch.argmax(pred, dim=1)
        target_indices = torch.argmax(target, dim=1)
        correct_predictions = torch.sum(predicted_indices == target_indices).item()
        accuracy = correct_predictions / len(predicted_indices)
        return loss, accuracy

    def log_metrics(self, loss, accuracy, prefix="train"):
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{prefix}_acc",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def log_confusion_matrix(self, conf_matrix):
        num_classes = conf_matrix.shape[0]
        # 设置标签，从 3 开始，以 3 的倍数递增
        labels = [
            f"{(i + 1) * 3}h" for i in range(num_classes)
        ]  # 生成标签，例如 '3h', '6h', ...

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xticks(np.arange(num_classes) + 0.5)  # 设置 x 轴刻度位置
        ax.set_yticks(np.arange(num_classes) + 0.5)  # 设置 y 轴刻度位置
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Time Lag")
        ax.set_ylabel("Actual Time Lag")
        ax.set_title("Confusion Matrix (Original)")
        plt.savefig("confusion_matrix_original.png")
        wandb.log(
            {"confusion_matrix_original": wandb.Image("confusion_matrix_original.png")}
        )
        plt.close()

        # 归一化后的混淆矩阵
        conf_matrix_normalized = (
            conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = sns.heatmap(
            conf_matrix_normalized,
            annot=False,
            cmap="Blues",
            norm=LogNorm(vmin=1e-2, vmax=1),  # 设置最小值和最大值来控制颜色范围
            ax=ax,
        )

        ax.set_xticks(np.arange(num_classes) + 0.5)  # 设置 x 轴刻度位置
        ax.set_yticks(np.arange(num_classes) + 0.5)  # 设置 y 轴刻度位置
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set_xlabel("Predicted Time Lag")
        ax.set_ylabel("Actual Time Lag")
        ax.set_title("Confusion Matrix (Normalized)")

        # 设置 color bar 的刻度
        colorbar = heatmap.collections[0].colorbar
        colorbar.set_ticks([1e-2, 1e-1, 1])  # 设置刻度
        colorbar.set_ticklabels(
            [r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
        )  # 设置刻度标签

        plt.savefig("confusion_matrix_normalized.png")
        wandb.log(
            {
                "confusion_matrix_normalized": wandb.Image(
                    "confusion_matrix_normalized.png"
                )
            }
        )
        plt.close()

# %% [markdown]
# ## Selection of Models and Datasets
# ### (time_interval, num_samples, exp_target; epoch)

# %%
## Create an AtmodistDataset instance
project_name = "Atmodist"  ## Atmodist, OrdinalAtmodist, TripletAtmodist
exp_target = "Formal"
exp_num = 4

time_interval = 45
num_samples = len(atmosphere_data)*3
epochs = 240

if project_name == "Atmodist":
    model = Atmodist(
        num_classes=time_interval // selected_frequency,
        res_in_channels=len(atmosphere_data[0][1]),
    )
    dataset = AtmodistDataset(
        data=atmosphere_data,
        num_samples=num_samples,
        selected_frequency=selected_frequency,
        time_unit=time_unit,
        time_interval=time_interval,
    )
elif project_name == "OrdinalAtmodist":
    model = OrdinalAtmodist(
        num_classes=time_interval // selected_frequency,
        res_in_channels=len(atmosphere_data[0][1]),
    )
    dataset = OrdinalDataset(
        data=atmosphere_data,
        num_samples=num_samples,
        selected_frequency=selected_frequency,
        time_unit=time_unit,
        time_interval=time_interval,
    )
elif project_name == "TripletAtmodist":
    solpos = pd.read_csv(
        "../dataset/solar_position.csv",
        index_col=0,
    )
    model = TripletAtmodist(
        res_in_channels=len(atmosphere_data[0][1]),
    )

    dataset = TripletDataset(
        data=atmosphere_data,
        solpos=solpos,
        num_samples=num_samples,
        time_unit=time_unit,
        time_range=time_interval * 24 * 3600,  # n days in seconds
        angle_limit=20,  # Lower angle difference for positive samples
        latitude=52,
        longitude=-2,
    )
else:
    print("Wrong Project Name !")

# %% [markdown]
# ## Dataset Processing (batch_size)  
# 

# %%
## Create a DataLoader to access the data
batch_size = 32
total_size = len(dataset)
train_set_size = int(0.8 * total_size)
valid_set_size = total_size - train_set_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_set_size, valid_set_size]
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
)
print(
    f"Original Dataset Size: {len(dataset)}, Training Set Size: {len(train_dataset)}, Validation Set Size: {len(val_dataset)}\n"
    f"Training DataLoader Batches: {len(train_loader)}, Validation DataLoader Batches: {len(val_loader)}"
)

# %% [markdown]
# ## Training
# 

# %%
model_notes = "trained"  # test, trained
model_name = f"{exp_target}-{exp_num}-{variables}-{selected_frequency}{time_unit}{time_interval}-{resample_method}-{preprocessing_method}"

model_directory = f"../../../models/{project_name}/{exp_target}"
checkpoint_dir = f"{model_directory}/checkpoints/"
model_dir = f"{model_directory}/{model_notes}/"
ensure_directory_exists(model_directory)
ensure_directory_exists(checkpoint_dir)
ensure_directory_exists(model_dir)
model_path = f"{model_dir}{model_name}.pth"
## wandb setup
###os.environ["WANDB_NOTEBOOK_NAME"] = "atmodist.ipynb"
###wandb.init(project=project_name, entity=user, name=model_name, notes=notes)
wandb.login(key=config["api_key"]["wandb"])
wandb_logger = WandbLogger(
    project=project_name, entity="dylan1120", name=model_name, log_model="all"
)

## Training setup
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    dirpath=checkpoint_dir,
    filename=f"{model_name}-{{epoch:02d}}-{{val_acc:.2f}}",
    save_top_k=1,
    mode="max",
)

trainer = Trainer(
    max_epochs=epochs + 1,
    logger=wandb_logger,
    check_val_every_n_epoch=1,
    callbacks=checkpoint_callback,
    accelerator="auto",
)
trainer.fit(model, train_loader, val_loader)

torch.save(model.state_dict(), model_path)
wandb.finish()

del model
del train_loader
del val_loader
del train_dataset
del val_dataset
del dataset
del atmosphere_data  # 删除 atmosphere_data 以释放内存
gc.collect()  # 强制垃圾回收

# %%



