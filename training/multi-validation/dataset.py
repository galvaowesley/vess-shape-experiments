from pathlib import Path

import torch
# noinspection PyUnresolvedReferences
from torchtrainer.datasets.vessel_base import DRIVE, VessMAP
from static_vess_shape_dataset import get_vess_shape_dataset # Static dataset
from vessel_shape_dataset.vessel_texture import VesselTexture # Dynamic dataset
from vessel_shape_dataset.vessel_dataset import normalize_img, VesselShapeDataset
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F

import torch
from PIL import Image
import numpy as np


class ValidTransforms:
    """Validation transform that only resizes the image."""

    def __init__(self, resize_size = None, resize_target = True):
        self.resize_size = resize_size
        self.resize_target = resize_target

    def __call__(self, img, target):

        img = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.uint8)
        target = torch.from_numpy(target).unsqueeze(0).to(dtype=torch.uint8)
        
        img = normalize_img(img, gray_scale=True)
        
        if self.resize_size is not None:
            img = tv_transf_F.resize(img, self.resize_size)
            if self.resize_target:
                target = tv_transf_F.resize(target, 
                                            self.resize_size, 
                                            interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        # img = img.float()/255
        target = target.to(dtype=torch.int64)
        target = target.squeeze(0)  # <-- Remover a dimensão extra

        return img, target

def get_datasets(datasets_root):

    ignore_index = 2  # For the DRIVE dataset
    root_vesshape = Path(datasets_root) / "VessShape/curves"
    root_drive = Path(datasets_root) / "DRIVE"
    root_vessmap = Path(datasets_root) / "VessMAP"

    transforms_256 = ValidTransforms(resize_size=(256, 256))
    transforms_512 = ValidTransforms(resize_size=(512, 512))

    size = 256
    n_control_points_range = (2, 20)    # "complexity" of the curves
    max_vd_range = (50.0, 150.0)        # Sets the typical curvature of the curves
    radius_range = (1, 5)               # Radius of each curve
    num_curves_range = (1, 20)          # Number of curves to generate
    sigma = (1, 2)                      # Standard deviation of the Gaussian noise 
        
    # Parameters for the dataset
    texture_dir = '/media/wesleygalvao/1_TB_LINUX/Datasets/ImageNet/ILSVRC2012_img_val/'  # Update as needed
    annotation_csv = '/media/wesleygalvao/1_TB_LINUX/Datasets/ImageNet/ILSVRC2012_img_val_annotation.csv'

    # Instanciando os geradores com ranges
    vessel_texture_train = VesselTexture(
        image_size=size,
        n_control_points=n_control_points_range,
        max_vd=max_vd_range,
        radius=radius_range,
        num_curves=num_curves_range,
        texture_dir=texture_dir,
        annotation_csv=annotation_csv,
        crop_size=(size, size),
        sigma=sigma
    )
    vessel_texture_valid = VesselTexture(
        image_size=size,
        n_control_points=n_control_points_range,
        max_vd=max_vd_range,
        radius=radius_range,
        num_curves=num_curves_range,
        texture_dir=texture_dir,
        annotation_csv=annotation_csv,
        crop_size=(size, size),
        sigma=sigma
    )
    
    ds_train_vessshape = VesselShapeDataset(vessel_texture_train, n_samples=50_000, gray_scale=True, normalize=True)
    
    _, ds_valid_vessshape, _, _, _ = get_vess_shape_dataset(
        dataset_path=root_vesshape, 
        mode="train",
        split_strategy="file", 
        resize_size=(256, 256), 
    )
        
    
    ds_valid_drive = DRIVE(root_drive, split="test", channels="gray", keepdim=True, 
                     ignore_index=ignore_index, transforms=transforms_512)
    ds_valid_vessmap = VessMAP(root_vessmap, keepdim=True, transforms=transforms_256)

    ds_valids = {
        "VessShape": ds_valid_vessshape,
        "DRIVE": ds_valid_drive,
        "VessMAP": ds_valid_vessmap
    }
    
    class_weights = (0.348, 0.652)
    # class_weights = (0.1036, 0.8964) # Dynamic VesselShape

    return ds_train_vessshape, ds_valids, class_weights, ignore_index


