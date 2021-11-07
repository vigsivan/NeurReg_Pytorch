"""
Trains the NeurReg model
"""

from tqdm import trange
from pathlib import Path
from dataset import ImageDataset
from components import *
from torch.nn import Conv3d, Sequential, Softmax
import torch

from losses import NeurRegLoss
from torch.utils.data import random_split, DataLoader

use_cuda = True

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
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

N = Unet3D(inshape=target_shape)
loss_func = NeurRegLoss(10,10)
stn = SpatialTransformer(target_shape)

to_flow_field = Conv3d(16,3, 3, padding=1)
conv_w_softmax = Sequential(Conv3d(17,1,3, padding=1), Softmax(3))

if use_cuda:
    N = N.cuda()
    to_flow_field = to_flow_field.cuda()
    conv_w_softmax = conv_w_softmax.cuda()

learnable_params = list(N.parameters())+ list(to_flow_field.parameters())+ list(conv_w_softmax.parameters())

optimizer = torch.optim.Adam(learnable_params, lr=1e-3)

data = dataset[0]

epochs = 1
for epoch in trange(epochs):
    for step, data in enumerate(train_loader):

        optimizer.zero_grad()

        if use_cuda:
            for category in ("moving", "transformed", "another"):
                for typ in ("image", "seg"):
                    data[category][typ] = data[category][typ].cuda()
                if category == "transformed":
                    data[category]["field"] = data[category]["field"].cuda()

        x1 = torch.cat((data["moving"]["image"], data["transformed"]["image"]), dim=1)
        x2 = torch.cat((data["moving"]["image"], data["another"]["image"]), dim=1)
        batched_images = torch.cat((x1, x2), dim=0)

        s1 = torch.cat((data["moving"]["seg"], data["transformed"]["seg"]), dim=1)
        s2 = torch.cat((data["moving"]["seg"], data["another"]["seg"]), dim=1)
        batched_segmentations = torch.cat((x1,x2), dim=0)

        last_layer = N(batched_images)
        last_layer0  = last_layer[0,:,:,:,:].unsqueeze(0)
        batched_fields = to_flow_field(last_layer)

        # Using notation from the paper
        F_0g = data["transformed"]["field"].squeeze().unsqueeze(0)
        F_0  = batched_fields[0,:,:,:,:].unsqueeze(0)
        F_1  = batched_fields[1,:,:,:,:].unsqueeze(0)
        I_0  = data["transformed"]["image"]
        I_1  = data["another"]["image"]
        I_0R = stn(data["moving"]["image"], F_0)
        I_1R = stn(data["moving"]["image"], F_1)
        S_0g = data["transformed"]["seg"]
        S_0  = stn(data["moving"]["seg"], F_0)
        S_1  = stn(data["moving"]["seg"], F_1)
        S_1g = stn(data["another"]["seg"], F_1)
        boosted = torch.cat((last_layer0, S_0), 1)
        S_0feat = conv_w_softmax(boosted)
        loss = loss_func(F_0, F_0g, I_0, I_0R, I_1, I_1R, S_0feat, S_0g, S_1, S_1g)
        loss.backward()
        optimizer.step()

# Steps:
# 1. Test losses
# 2. Test cuda
# 3. Deploy!
