# Standard library imports
import copy
import datetime
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

# Wandb (Weights & Biases) related imports
import wandb


def batch_data_parser(data_loader, message=True):
    batch_data = next(iter(data_loader))
    if message:
        print(f"Batch data type: {type(batch_data)}")
        if isinstance(batch_data, (list, tuple)):
            print(f"Number of elements in the batch: {len(batch_data)}")
            for i, data in enumerate(batch_data):
                if hasattr(data, "shape"):
                    print(f"Element {i}: Type={type(data)}, Shape={data.shape}")
                else:
                    print(f"Element {i}: Type={type(data)}, Value={data}")
    else:
        pass
    return batch_data  # timestamp1, timestamp2, images1, images2, labels


# timestamp1, timestamp2, images1, images2, labels = batch_data_parser(train_loader)


def model_parser(module, checkpoint_path, message=True):
    model = module.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(torch.device("cpu"))
    if message:
        for i, stage in enumerate(model.children()):
            print(f"Layer {i}: {stage}")
    else:
        pass
    return model

"""class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        children = list(model.children())
        specific_stage = children[0]
        self.part_of_model = specific_stage

    def forward(self, x):
        return self.part_of_model(x)

model_checkpoint_path = "/home/yuhan/Desktop/Master/models/Atmodist/ModelStructure/checkpoints/ModelStructure-d2muvmslr-3h69-mean_weighted_standardized-epoch=00-val_loss=2.99.ckpt"
model = model_parser(Atmodist, model_checkpoint_path, False)
model = ModelWrapper(model)"""


def feature_map_parser(model, input):
    def print_shape_hook(module, input, output):
        if isinstance(output, tuple):
            print(
                f"{module.__class__.__name__} output shapes: {[o.shape for o in output]}"
            )
        else:
            print(f"{module.__class__.__name__} output shape: {output.shape}")

    hook_handles = []
    for name, layer in model.named_modules():
        handle = layer.register_forward_hook(print_shape_hook)
        hook_handles.append(handle)
    # copied_model = copy.deepcopy(model)
    model(input)
    for handle in hook_handles:
        handle.remove()


# feature_map_parser(model, (images1[:16], images2[:16]))
"""batch = next(iter(train_loader))
some_input_tensor = batch[2]  
feature_map_parser(model, (some_input_tensor,some_input_tensor))"""



def process_in_batches(model, data, batch_size=1000, output_path="results.npy"):
    # Check if the file exists and delete it
    if os.path.exists(output_path):
        os.remove(output_path)

    num_samples = data.shape[0]
    results = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_data = data[start:end]
        
        batch_result = model(batch_data)
        batch_result = batch_result.detach().numpy()
        results.append(batch_result)
        
        if len(results) >= 10: 
            with open(output_path, 'ab') as f:
                np.save(f, np.concatenate(results, axis=0))
            results = [] 
        
    if results:
        with open(output_path, 'ab') as f:
            np.save(f, np.concatenate(results, axis=0))

def read_results(output_path="results.npy"):
    results = []
    with open(output_path, 'rb') as f:
        while True:
            try:
                results.append(np.load(f))
            except EOFError:
                break
    return np.concatenate(results, axis=0)



"""batch_size = 1000
process_in_batches(model, atmosphere_data_without_timestamps, batch_size)
embedding_results = read_results()
print(embedding_results.shape)
embedding_results = embedding_results.reshape(embedding_results.shape[0], -1)
print(embedding_results.shape)"""