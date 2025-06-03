import os
import datetime
import numpy as np
import pandas as pd
import random
import math
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn import functional as F
import torchmetrics
import pvlib


# --- Utility Functions ---

def one_hot_encode_indexes(delta_t_list, num_classes, selected_frequency):
    delta_t_list = [(delta_t // selected_frequency) - 1 for delta_t in delta_t_list]
    indexes = torch.tensor(delta_t_list)
    one_hot_indexes = torch.nn.functional.one_hot(indexes, num_classes=num_classes).to(torch.float32)
    return one_hot_indexes


def ordinal_encode_indexes(delta_t_list, num_classes, selected_frequency):
    delta_t_list = [delta_t // selected_frequency for delta_t in delta_t_list]
    indexes = torch.tensor(delta_t_list)
    return indexes


def calculate_time_difference(img1_timestamp, img2_timestamp, time_unit):
    try:
        delta_time = (
            abs(img1_timestamp - img2_timestamp)
            .astype(f"timedelta64[{time_unit}]")
            .astype("int32")
        )
        return int(delta_time)
    except Exception as e:
        raise ValueError(f"Error calculating time difference: {e}")


def get_timestamp(img):
    try:
        timestamp = np.datetime64(img[0], "s").astype(int)
        return timestamp
    except Exception as e:
        raise ValueError(f"Error extracting timestamp: {e}")


# --- Dataset Classes ---

class AtmodistDatasetUncovered(Dataset):
    """
    Dataset that generates unique sample pairs with time delta labels (one-hot).
    """
    def __init__(self, data, num_samples, selected_frequency, time_unit, time_interval):
        self.sample_pairs = []
        self.time_deltas = []
        self.timestamps_1 = []
        self.timestamps_2 = []
        self.num_variables = len(data[0][1])
        num_classes = time_interval // selected_frequency

        self.data = data
        self.timestamps = [item[0] for item in data]
        num_timestamps = len(self.timestamps)
        used_pairs = set()

        max_steps = time_interval // selected_frequency
        attempts = 0
        max_attempts = num_samples * 10  # Prevent infinite loop

        while len(self.sample_pairs) < num_samples and attempts < max_attempts:
            img1_index = np.random.choice(num_timestamps)
            random_step = np.random.randint(1, max_steps + 1)
            img2_index = np.clip(img1_index + random_step, 1, num_timestamps - 1)

            pair_key = tuple(sorted((img1_index, img2_index)))
            if pair_key in used_pairs:
                attempts += 1
                continue

            img1_timestamp, img1_data = self.data[img1_index]
            img2_timestamp, img2_data = self.data[img2_index]

            try:
                time_difference = calculate_time_difference(img1_timestamp, img2_timestamp, time_unit)
            except Exception:
                attempts += 1
                continue

            if 0 < time_difference <= time_interval:
                self.sample_pairs.append((img1_data, img2_data))
                self.time_deltas.append(time_difference)
                self.timestamps_1.append(np.datetime64(img1_timestamp, "s").astype(int))
                self.timestamps_2.append(np.datetime64(img2_timestamp, "s").astype(int))
                used_pairs.add(pair_key)
            attempts += 1

        self.deltat_labels = one_hot_encode_indexes(self.time_deltas, num_classes, selected_frequency)

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, index):
        x1, x2 = self.sample_pairs[index]
        r1 = torch.from_numpy(np.array(x1).astype(np.float32))
        r2 = torch.from_numpy(np.array(x2).astype(np.float32))
        return [
            self.timestamps_1[index],
            self.timestamps_2[index],
            r1,
            r2,
            self.deltat_labels[index],
        ]


class AtmodistDataset(Dataset):
    """
    Dataset that generates random sample pairs with time delta labels (one-hot).
    """
    def __init__(self, data, num_samples, selected_frequency, time_unit, time_interval):
        self.sample_pairs = []
        self.time_deltas = []
        self.timestamps_1 = []
        self.timestamps_2 = []
        self.num_variables = len(data[0][1])
        num_classes = time_interval // selected_frequency

        self.data = data
        self.timestamps = [item[0] for item in data]
        num_timestamps = len(self.timestamps)

        max_steps = time_interval // selected_frequency
        random_indices_1 = np.random.choice(num_timestamps, num_samples)
        random_steps = np.random.randint(1, max_steps + 1, num_samples)
        random_indices_2 = np.clip(random_indices_1 + random_steps, 1, num_timestamps - 1)

        for img1_index, img2_index in zip(random_indices_1, random_indices_2):
            img1_timestamp, img1_data = self.data[img1_index]
            img2_timestamp, img2_data = self.data[img2_index]
            try:
                time_difference = calculate_time_difference(img1_timestamp, img2_timestamp, time_unit)
            except Exception:
                continue

            if 0 < time_difference <= time_interval:
                self.sample_pairs.append((img1_data, img2_data))
                self.time_deltas.append(time_difference)
                self.timestamps_1.append(np.datetime64(img1_timestamp, "s").astype(int))
                self.timestamps_2.append(np.datetime64(img2_timestamp, "s").astype(int))

        self.deltat_labels = one_hot_encode_indexes(self.time_deltas, num_classes, selected_frequency)

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, index):
        x1, x2 = self.sample_pairs[index]
        r1 = torch.from_numpy(np.array(x1).astype(np.float32))
        r2 = torch.from_numpy(np.array(x2).astype(np.float32))
        return [
            self.timestamps_1[index],
            self.timestamps_2[index],
            r1,
            r2,
            self.deltat_labels[index],
        ]


class OrdinalDataset(Dataset):
    """
    Dataset that generates random sample pairs with ordinal time delta labels.
    """
    def __init__(self, data, num_samples, selected_frequency, time_unit, time_interval):
        self.sample_pairs = []
        self.time_deltas = []
        self.timestamps_1 = []
        self.timestamps_2 = []
        self.num_variables = len(data[0][1])
        num_classes = time_interval // selected_frequency

        self.data = data
        self.timestamps = [item[0] for item in data]
        num_timestamps = len(self.timestamps)

        max_steps = time_interval // selected_frequency
        random_indices_1 = np.random.choice(num_timestamps, num_samples)
        random_steps = np.random.randint(1, max_steps + 1, num_samples)
        random_indices_2 = np.clip(random_indices_1 + random_steps, 0, num_timestamps - 1)

        for img1_index, img2_index in zip(random_indices_1, random_indices_2):
            img1_timestamp, img1_data = self.data[img1_index]
            img2_timestamp, img2_data = self.data[img2_index]
            try:
                time_difference = calculate_time_difference(img1_timestamp, img2_timestamp, time_unit)
            except Exception:
                continue

            if time_difference < time_interval:
                self.sample_pairs.append((img1_data, img2_data))
                self.time_deltas.append(time_difference)
                self.timestamps_1.append(np.datetime64(img1_timestamp, "s").astype(int))
                self.timestamps_2.append(np.datetime64(img2_timestamp, "s").astype(int))

        self.deltat_labels = ordinal_encode_indexes(self.time_deltas, num_classes, selected_frequency)

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, index):
        x1, x2 = self.sample_pairs[index]
        r1 = torch.from_numpy(np.array(x1).astype(np.float32))
        r2 = torch.from_numpy(np.array(x2).astype(np.float32))
        return [
            self.timestamps_1[index],
            self.timestamps_2[index],
            r1,
            r2,
            self.deltat_labels[index],
        ]


class TripletDataset(Dataset):
    """
    Dataset for triplet loss training, using solar elevation angle as positive/negative criterion.
    """
    def __init__(
        self,
        data,
        solpos,
        num_samples,
        time_range,
        angle_limit,
        latitude=52,
        longitude=-2,
    ):
        self.data = data
        self.solpos = solpos
        self.num_samples = num_samples
        self.time_range = time_range
        self.angle_limit = angle_limit
        self.latitude = latitude
        self.longitude = longitude
        self.triplets = self.generate_triplets()

    def solar_angle_difference(self, img1, img2):
        try:
            elevation1 = int(self.solpos.loc[get_timestamp(img1)]["elevation"])
            elevation2 = int(self.solpos.loc[get_timestamp(img2)]["elevation"])
            return abs(elevation1 - elevation2)
        except Exception as e:
            raise ValueError(f"Error in solar_angle_difference: {e}")

    def generate_triplets(self):
        triplets = []
        attempts = 0
        max_attempts = self.num_samples * 10
        while len(triplets) < self.num_samples and attempts < max_attempts:
            anchor_idx = random.randint(0, len(self.data) - 1)
            anchor = self.data[anchor_idx]
            anchor_ts = get_timestamp(anchor)

            # Candidates within time_range
            candidates = [
                item for item in self.data
                if 0 < abs(get_timestamp(item) - anchor_ts) <= self.time_range
            ]
            positive_candidates = [
                item for item in candidates
                if self.solar_angle_difference(anchor, item) <= self.angle_limit
            ]
            negative_candidates = [
                item for item in candidates
                if self.solar_angle_difference(anchor, item) > self.angle_limit
            ]

            if positive_candidates and negative_candidates:
                positive = random.choice(positive_candidates)
                negative = random.choice(negative_candidates)
                triplets.append((anchor, positive, negative))
            attempts += 1
        if len(triplets) < self.num_samples:
            print(f"Warning: Only generated {len(triplets)} triplets out of {self.num_samples} requested.")
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        anchor_tensor = torch.tensor(anchor[1], dtype=torch.float32)
        positive_tensor = torch.tensor(positive[1], dtype=torch.float32)
        negative_tensor = torch.tensor(negative[1], dtype=torch.float32)
        return anchor_tensor, positive_tensor, negative_tensor
