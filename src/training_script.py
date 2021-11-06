"""
Trains the NeurReg model
"""

from tqdm import trange
from pathlib import Path
from dataset import ImageDataset
from components import *
from torch.nn import Conv3d
import torch

from losses import NeurRegLoss
from torch.utils.data import random_split, DataLoader


path_to_images = "/Volumes/Untitled/Task04_Hippocampus/imagesTr/"
path_to_segs = "/Volumes/Untitled/Task04_Hippocampus/labelsTr/"
matching_fn = lambda x: x

target_shape = (128, 128, 128)
dataset = ImageDataset(
    Path(path_to_images), Path(path_to_segs), matching_fn, target_shape=target_shape
)

train_proportion = 0.9
train_len = int(train_proportion * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

N = Unet3D(inshape=target_shape)
optimizer = torch.optim.Adam(N.parameters(), lr=1e-3)
loss = NeurRegLoss(10,10)
stn = SpatialTransformer(target_shape)

to_flow_field = Conv3d(16,3, 3, padding=1)

epochs = 1000
for epoch in trange(epochs):
    for step, data in enumerate(train_loader):
        x1 = torch.cat((data["moving"]["image"], data["transformed"]["image"]), dim=1)
        x2 = torch.cat((data["moving"]["image"], data["another"]["image"]), dim=1)
        batched_images = torch.cat((x1, x2), dim=0)

        s1 = torch.cat((data["moving"]["seg"], data["transformed"]["seg"]), dim=1)
        s2 = torch.cat((data["moving"]["seg"], data["another"]["seg"]), dim=1)
        batched_segmentations = torch.cat((x1,x2), dim=0)

        last_layer = N(batched_images)
        batched_fields = to_flow_field(last_layer)


