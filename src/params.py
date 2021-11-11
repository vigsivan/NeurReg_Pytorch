import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CPU_CONFIG:
    #########################
    # Set to True when training
    # FIXME: better to use device rather than boolean (more clean logic)
    device = "cpu"

    #########################
    # How much should be used for training
    train_proportion = 1.0

    #########################
    # Spatial shape of each tensor that goes throgh the network
    target_shape = (128, 128, 128)

    #########################
    # Path to the data
    data_root_dir = Path("/Volumes/Untitled/Task04_Hippocampus/")
    path_to_images = data_root_dir / "imagesTr/"
    path_to_segs = data_root_dir / "labelsTr/"

    #########################
    # Function for matching image and seg files
    matching_fn = lambda x: x

    #########################
    # Number of workers
    num_workers = 3

    #########################
    # Loss params weighting
    cross_corr_loss_weight, seg_loss_weight = 10, 10

    #########################
    # Number of epochs
    epochs = 1

    #########################
    # Number of epochs
    lr = 1e-3

    #########################
    # Path to save stuff
    savedir = Path(".")
    checkpoint = savedir / "checkpoint.pt"
    step_loss_file = savedir / "step_loss.txt"
    epochs_per_save = 2


@dataclass
class SLURM_CONFIG:
    #########################
    # Set to True when training
    # FIXME: better to use device rather than boolean (more clean logic)
    device = "cuda"

    #########################
    # How much should be used for training
    train_proportion = 1.0

    #########################
    # Spatial shape of each tensor that goes throgh the network
    target_shape = (128, 128, 128)

    #########################
    # Path to the data
    data_root_dir = Path("//localscratch/vsivan.27564466.0/Task04_Hippocampus/")
    path_to_images = data_root_dir / "imagesTr/"
    path_to_segs = data_root_dir / "labelsTr/"

    #########################
    # Function for matching image and seg files
    matching_fn = lambda x: x

    #########################
    # Number of workers
    num_workers = 12

    #########################
    # Loss params weighting
    cross_corr_loss_weight, seg_loss_weight = 10, 10

    #########################
    # Number of epochs
    epochs = 1000

    #########################
    # Number of epochs
    lr = 1e-3

    #########################
    # Path to save stuff
    savedir = Path("/home/vsivan/scratch/NeurReg_Pytorch/logging")
    checkpoint = savedir / "checkpoint.pt"
    step_loss_file = savedir / "step_loss.txt"
    epochs_per_save = 2
