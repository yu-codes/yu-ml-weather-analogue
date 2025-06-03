# Standard library imports
import math

# PyTorch related imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# PyTorch Lightning related imports
import pytorch_lightning as pl

class ShortcutDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, downscale=False):
        super(ShortcutDown, self).__init__()

        self.out_channels = out_channels
        self.stride = 2 if downscale else 1
        padding = math.floor((kernel_size - 1) / 2)

        # If the input channels are different from output channels, apply convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            bias=False,
        )

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
                downscale = (block_idx == 0 or block_idx == 1) and (idx == 0)
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
    ):
        super(Encoder, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=out_conv_channels,
            kernel_size=conv_kernel_size,
            padding=math.floor((conv_kernel_size - 1) / 2),
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            maxpool_kernel_size,
            stride=stride,
            padding=math.floor((maxpool_kernel_size - 1) / 2),
        )

        # Resblocks
        self.resnet = ResNet(out_conv_channels, channel_number_list)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)
        return x


class TripletEncoder(nn.Module):

    def __init__(
        self,
        in_channels,
        channel_number_list,
        out_conv_channels=16,
        conv_kernel_size=8,
        maxpool_kernel_size=3,
        stride=2,
        embedding_dim=128,
    ):
        super(TripletEncoder, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=out_conv_channels,
            kernel_size=conv_kernel_size,
            padding=math.floor((conv_kernel_size - 1) / 2),
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            maxpool_kernel_size,
            stride=stride,
            padding=math.floor((maxpool_kernel_size - 1) / 2),
        )

        # Resblocks
        self.resnet = ResNet(out_conv_channels, channel_number_list)

        # Embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, embedding_dim)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class Atmodist(pl.LightningModule):

    def __init__(
        self,
        num_classes,
        res_in_channels,
        res_out_channels_list=(16, 32, 64, 128),
        downstream_stack_channels=256,
        downstream_channels=128,
        downstream_kernel_size=3,
        downstream_fin_shape=4,
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
        self.flat = nn.Flatten()
        self.linear = nn.Linear(
            128,
            # downstream_channels * downstream_fin_shape * downstream_fin_shape,
            self.num_classes,
            bias=False,
        )
        # self.softmax = nn.Softmax(
        #     dim=-1
        # )  ## Remove the softmax layer because nn.CrossEntropyLoss already includes softmax
        # Loss function and hyperparameters
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
        x = self.linear(x)
        # x = self.softmax(x)
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


class OrdinalAtmodist(pl.LightningModule):

    def __init__(
        self,
        num_classes,
        res_in_channels,
        res_out_channels_list=(16, 32, 64, 128),
        downstream_stack_channels=256,
        downstream_channels=128,
        downstream_kernel_size=3,
        downstream_fin_shape=4,
        lr=0.01,
        momentum=0.9,
    ):
        super(OrdinalAtmodist, self).__init__()
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
        self.flat = nn.Flatten()
        self.linear = nn.Linear(
            128,
            # downstream_channels * downstream_fin_shape * downstream_fin_shape,
            1,
            bias=False,
        )
        # 使用适合回归任务的损失函数
        self.loss = nn.MSELoss()  # Mean Squared Error Loss
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
        x = self.linear(x)
        # x = self.softmax(x)
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
        loss, accuracy = self.calculate_loss_and_accuracy(pred, target, threshold=1.0)
        self.log_metrics(loss, accuracy, prefix="train")

        return {"step": self.global_step, "loss": loss, "acc": accuracy}

    def validation_step(self, batch, batch_idx):
        timestamp1, timestamp2, r1, r2, target = batch
        atm1, atm2, pred = self.forward((r1, r2))
        loss, accuracy = self.calculate_loss_and_accuracy(pred, target, threshold=1.0)
        self.log_metrics(loss, accuracy, prefix="val")
        return {"val_loss": loss, "val_acc": accuracy}

    def calculate_loss_and_accuracy(self, pred, target, threshold=1.0):
        loss = self.loss(pred.squeeze(), target.float())
        abs_diff = torch.abs(pred.squeeze() - target.float())
        accurate_predictions = abs_diff < threshold
        accuracy = torch.mean(accurate_predictions.float())
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


class TripletAtmodist(pl.LightningModule):

    def __init__(
        self,
        res_in_channels,
        res_out_channels_list=(16, 32, 64, 128),
        embedding_dim=128,
        lr=0.01,
        momentum=0.9,
        margin=1.0,
    ):
        super(TripletAtmodist, self).__init__()
        self.validation_step_outputs = []

        # Models
        self.embedding = TripletEncoder(
            res_in_channels, res_out_channels_list, embedding_dim=embedding_dim
        )
        self.lr = lr
        self.momentum = momentum
        self.margin = margin
        self.save_hyperparameters()

    def forward(self, x):
        return self.embedding(x)

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

    def calculate_loss_and_accuracy(self, anchor, positive, negative):
        anchor_embed = self(anchor)
        positive_embed = self(positive)
        negative_embed = self(negative)

        distance_positive = (anchor_embed - positive_embed).pow(2).sum(1)
        distance_negative = (anchor_embed - negative_embed).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        loss = losses.mean()

        accuracy = (distance_positive < distance_negative).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        loss, acc = self.calculate_loss_and_accuracy(anchor, positive, negative)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"step": self.global_step, "loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        anchor, positive, negative = batch
        loss, acc = self.calculate_loss_and_accuracy(anchor, positive, negative)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss, "val_acc": acc}

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

    def compute_distance(self, x1, x2):
        embedding1 = self.forward(x1)
        embedding2 = self.forward(x2)
        distance = torch.norm(embedding1 - embedding2, p=2, dim=1)
        return distance
