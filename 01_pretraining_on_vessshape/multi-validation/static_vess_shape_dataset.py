import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def simple_img_opener_RGB(img_path):
    """
    Loads and returns an image as a tensor CxHxW (RGB).

    Args:
        img_path (str | Path): Path to the image file.

    Returns:
        torch.Tensor: Tensor of the image in the format [C, H, W].
    """
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img = np.array(img_pil, dtype=np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        raise e
    
def simple_img_opener(img_path):
    """
    Loads and returns an image as a tensor 1xHxW (grayscale).

    Args:
        img_path (str | Path): Path to the image file.

    Returns:
        torch.Tensor: Tensor of the image in the format [1, H, W].
    """
    try:
        img_pil = Image.open(img_path).convert("L")
        img = np.array(img_pil, dtype=np.uint8)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        return img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        raise e


def simple_label_opener(label_path):
    """
    Loads and returns a monochromatic label as a tensor HxW,
    where 0=background, 1=class.

    Args:
        label_path (str | Path): Path to the label file.

    Returns:
        torch.Tensor: Tensor of the label in the format [H, W].
    """
    try:
        lbl_pil = Image.open(label_path).convert("L")
        lbl_arr = np.array(lbl_pil, dtype=np.uint8)
        lbl_arr = lbl_arr // 255  # Convert 255 -> 1
        lbl = torch.from_numpy(lbl_arr).long()  # [H, W]
        return lbl
    except Exception as e:
        print(f"Error loading label {label_path}: {e}")
        raise e


class VessShapeDataset(torch.utils.data.Dataset):
    """
    Minimalistic dataset for 'vess_shape'.

    Expected structure:
      - <dataset_path>/images: Images (JPG or PNG).
      - <dataset_path>/labels: Corresponding labels (PNG).
    """

    def __init__(self, img_files, lbl_files, transform=None):
        """
        Initializes the dataset with lists of image and label files.

        Args:
            img_files (list of str | Path): List of image file paths.
            lbl_files (list of str | Path): List of label file paths.
            transform (callable, optional): Transformation function to be applied to the data.
        """
        super().__init__()
        self.img_files = img_files
        self.lbl_files = lbl_files
        self.img_files = sorted(img_files)
        self.lbl_files = sorted(lbl_files)
        self.images = self.img_files
        self.labels = self.lbl_files
        self.transform = transform

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label)
        """
        img = simple_img_opener(self.img_files[idx])
        lbl = simple_label_opener(self.lbl_files[idx])
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def split_train_val(self, val_frac=0.1, seed=0):
        """
        Randomly splits the dataset into training and validation.

        Args:
            val_frac (float, optional): Fraction of the data for validation. Default is 0.1.
            seed (int, optional): Seed for the random number generator. Default is 0.

        Returns:
            tuple: (ds_train, ds_valid)
        """
        n = len(self)
        n_val = int(n * val_frac)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        idx_train = indices[n_val:]
        idx_val = indices[:n_val]

        train_imgs = [self.img_files[i] for i in idx_train]
        train_lbls = [self.lbl_files[i] for i in idx_train]
        valid_imgs = [self.img_files[i] for i in idx_val]
        valid_lbls = [self.lbl_files[i] for i in idx_val]

        ds_train = VessShapeDataset(train_imgs, train_lbls, transform=None)
        ds_valid = VessShapeDataset(valid_imgs, valid_lbls, transform=None)
        return ds_train, ds_valid

    def set_transform(self, transform):
        """
        Sets the transformation to be applied to the data.

        Args:
            transform (callable): Transformation function.
        """
        self.transform = transform


def get_vess_shape_dataset(
    dataset_path, 
    mode="train",
    split_strategy="file", 
    resize_size=(256, 256), 
    **kwargs
):
    """
    Loads and returns the 'vess_shape' dataset divided into training and validation.

    Parameters
    ----------
    dataset_path : str | Path
        Path to the root directory of the dataset.
    split_strategy : str, optional
        Strategy to split the dataset. 
        Options:
            - "rand_<frac>": Uses <frac> fraction of data for validation.
            - "file": Uses train.csv and val.csv files to split the dataset.
        Default is "rand_0.2".
    resize_size : tuple of int, optional
        Size to resize the images. Default is (256, 256).
    **kwargs : dict
        Additional parameters, such as 'seed' for reproducibility.

    Returns
    -------
    tuple
        ds_train (VessShapeDataset): Training dataset.
        ds_valid (VessShapeDataset): Validation dataset.
        class_weights (tuple of float): Class weights.
        ignore_index (int | None): Index to ignore in labels.
        collate_fn (callable | None): Collate function for the DataLoader.
    """
    dataset_path = Path(dataset_path)
    img_dir = dataset_path / "images"
    lbl_dir = dataset_path / "labels"

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    lbl_files = sorted(lbl_dir.glob("*.png"))

    print(f"Found {len(img_files)} image files.")
    print(f"Found {len(lbl_files)} label files.")

    if len(img_files) != len(lbl_files):
        raise ValueError(f"Number of images ({len(img_files)}) and labels ({len(lbl_files)}) do not match.")

    ds = VessShapeDataset(img_files, lbl_files)

    if "rand" in split_strategy:
        split_frac = float(split_strategy.split("_")[1])
        print(f"Using random split strategy with validation fraction: {split_frac}")
        ds_train, ds_valid = ds.split_train_val(val_frac=split_frac, seed=kwargs.get("seed", 0))
        print(f"Training samples: {len(ds_train)}")
        print(f"Validation samples: {len(ds_valid)}")
    elif split_strategy == "file":
        train_csv = dataset_path / "train.csv"
        val_csv = dataset_path / "val.csv"
        test_csv = dataset_path / "test.csv"
        print(f"Using file-based split strategy with train.csv, val.csv and test.csv.")
        if not train_csv.exists() or not val_csv.exists():
            raise FileNotFoundError("train.csv or val.csv or test.csv not found in the dataset directory.")

        with open(train_csv) as file:
            files_train = file.read().splitlines()
        with open(val_csv) as file:
            files_valid = file.read().splitlines()
        with open(test_csv) as file:
            files_test = file.read().splitlines()

        ds_train = VessShapeDataset(
            img_files=[dataset_path / "images" / fname for fname in files_train],
            lbl_files=[dataset_path / "labels" / fname for fname in files_train],
            transform=None
        )
        ds_valid = VessShapeDataset(
            img_files=[dataset_path / "images" / fname for fname in files_valid],
            lbl_files=[dataset_path / "labels" / fname for fname in files_valid],
            transform=None
        )
        ds_test = VessShapeDataset(
            img_files=[dataset_path / "images" / fname for fname in files_test],
            lbl_files=[dataset_path / "labels" / fname for fname in files_test],
            transform=None
        )
        print(f"Training samples: {len(ds_train)}")
        print(f"Validation samples: {len(ds_valid)}")
        print(f"Test samples: {len(ds_test)}")
    else:
        raise ValueError(f"Split strategy '{split_strategy}' not recognized.")

    # Define mean and std for normalization
    # mean = torch.tensor([0.47456726, 0.44373271, 0.39307836]).reshape(3, 1, 1)
    # std = torch.tensor([0.23787807, 0.2321209, 0.23344506]).reshape(3, 1, 1)
    
    mean = torch.tensor([0.43712611]).reshape(1, 1, 1)
    std = torch.tensor([0.23448134]).reshape(1, 1, 1)

    def train_transform(img, lbl):
        """
        Training transformation: normalize image with channel-wise mean and std.

        Args:
            img (torch.Tensor): Image tensor [C, H, W].
            lbl (torch.Tensor): Label tensor [H, W].

        Returns:
            tuple: (normalized image, label)
        """
        img = img.float() / 255.0
        img = (img - mean) / std
        return img, lbl

    def valid_transform(img, lbl):
        """
        Validation transformation: normalize image with channel-wise mean and std.

        Args:
            img (torch.Tensor): Image tensor [C, H, W].
            lbl (torch.Tensor): Label tensor [H, W].

        Returns:
            tuple: (normalized image, label)
        """
        img = img.float() / 255.0
        img = (img - mean) / std
        return img, lbl

    print("Setting transformations.")
    ds_train.set_transform(train_transform)
    ds_valid.set_transform(valid_transform)
    ds_test.set_transform(valid_transform)

    # Define class weights and ignore_index as needed
    class_weights = (0.348, 0.652)  # Adjust based on dataset statistics
    ignore_index = None            # Example: 255, -100, etc.
    collate_fn = None              # Implement if necessary

    print("Dataset built successfully.") 
    
    if mode == "train":
        return ds_train, ds_valid, class_weights, ignore_index, collate_fn
    else:
        return ds_test, class_weights, ignore_index, collate_fn, None
