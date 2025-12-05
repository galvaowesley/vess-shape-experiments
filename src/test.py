"""
Reference script to test a trained model.
Please refer to the get_parser function to see the available arguments.
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image
import pandas as pd
import torch
import torch.utils
import yaml
from torchvision.transforms.v2 import functional as tv_transf_F
from tqdm.auto import tqdm

from torchtrainer.metrics.confusion_metrics import ConfusionMatrixMetrics, ROCAUCScore
from torchtrainer.util import test_util
from torchtrainer.util.train_util import ParseKwargs, WrapDict, dict_to_argv, seed_all

import segmentation_models_pytorch as smp



@torch.no_grad()
def test(param_dict=None):
    """Test a trained model. Please refer to the get_parser function to
    see the available arguments.
    """

    args = get_args(param_dict)

    # Configure matplotlib backend early if headless flag is set (flag parsed but not yet used)
    if getattr(args, 'force_headless', False):
        # Use Agg backend to avoid any Tkinter / X11 requirements
        import matplotlib
        matplotlib.use('Agg')

    seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = args.benchmark
    torch.set_float32_matmul_precision("high")

    device = args.device
    run_path = Path(args.run_path)

    # Setup the dataset and related elements.

    dataset_class = args.dataset_class
    dataset_path = args.dataset_path
    resize_size = args.resize_size
    dataset_params = args.dataset_params
    encoder_weights = args.encoder_weights
    
    if dataset_class == "oxford_pets":
        from torchtrainer.datasets.oxford_pets import get_dataset

        _, ds_test, *dataset_props = get_dataset(dataset_path, 0.2, resize_size)
        class_weights, ignore_index, _ = dataset_props
    elif dataset_class == "drive":
        from torchtrainer.datasets.vessel import get_dataset_drive_test

        ds_train, ds_test, *dataset_props = get_dataset_drive_test(
            dataset_path, "default", resize_size=resize_size, channels='gray', **dataset_params)
        class_weights, ignore_index = dataset_props
    elif dataset_class == "vessmap":
        from torchtrainer.datasets.vessel import get_dataset_vessmap_test

        # Warning, are relying on the seed to get the same dataset split used during training
        ds_train, ds_test, *dataset_props = get_dataset_vessmap_test(dataset_path)
        class_weights, ignore_index, _ = dataset_props
    
    elif dataset_class in {"dca1", "dca"}:
        from torchtrainer.datasets.vessel import get_dataset_dca1_test

        # Warning, are relying on the seed to get the same dataset split used during training
        ds_test, class_weights, *dataset_props = get_dataset_dca1_test(dataset_path, resize_size=resize_size)
        ignore_index, collate_fn = dataset_props
        if dataset_class == "dca":
            print("[WARN] dataset_class 'dca' is deprecated; use 'dca1' for clarity.")
    else:
        raise ValueError(
            f"Unsupported dataset_class='{dataset_class}'. Expected one of: oxford_pets, drive, vessmap, dca1"
        )

    num_classes = len(class_weights)
    num_channels = ds_test[0][0].shape[0]

    model_class = args.model_class
    model_params = args.model_params

    seed_all(args.seed)

    if model_class == "simple_encoder_decoder":
        from torchtrainer.models.simple_encoder_decoder import get_model

        model = get_model(**model_params, num_classes=num_classes)
    elif model_class == "unet_lw":
        from torchtrainer.models.unet_lw import get_model

        model = get_model(num_channels=num_channels, num_classes=num_classes)
    elif model_class == "test_model":
        from torchtrainer.models.testing import TestSegmentation

        model = TestSegmentation(num_channels=num_channels, num_classes=num_classes)
    elif model_class == "resnet18_unet":

        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,  # imagenet or None
            in_channels=num_channels,
            classes=num_classes,
        )
    elif model_class == "resnet50_unet":
        
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,  
            in_channels=num_channels,
            classes=num_classes,
        )
    else:
        raise NotImplementedError(f"Model '{model_class}' not implemented.")
    
    if not getattr(args, "skip_checkpoint_loading", False):
        try:
            if args.checkpoint_type == "best":
                checkpoint = torch.load(run_path/"best_model.pt", weights_only=False)
            elif args.checkpoint_type == "last":
                checkpoint = torch.load(run_path/"checkpoint.pt", weights_only=False)
            model.load_state_dict(checkpoint["model"])
        except FileNotFoundError as e:
            print(f"[WARN]: Could not load checkpoint: {e}. Proceeding with random weights.")
    else:        
        print("[INFO]: Skipping checkpoint load. Using encoder_weights or random decoder.")
    
    model.to(device)
    model.eval()
     
    # Load the model weights  
    # if args.checkpoint_type == "best":
    #     checkpoint = torch.load(run_path/"best_model.pt", weights_only=False)
    # elif args.checkpoint_type == "last":
    #     checkpoint = torch.load(run_path/"checkpoint.pt", weights_only=False)
# 
    # model.load_state_dict(checkpoint["model"])
    # model.to(device)
    # model.eval()

    # Test-time augmentation transforms
    transforms = (
        test_util.Rotation(0),
        test_util.Rotation(1),
        test_util.Rotation(2),
        test_util.Rotation(3),
        test_util.Flip(1),
        test_util.Flip(2),
        test_util.ReflectDiag(True),
        test_util.ReflectDiag(False),
        test_util.Scale(resize_size, 4.7),
        test_util.Scale(resize_size, 4.7, scale_up=False),
    )

    threshold = args.threshold
    if threshold == -1:
        threshold = test_util.find_optimal_threshold(
            model, ds_train, ignore_index=ignore_index, device=device)

    # Performance scores
    conf_metrics = ConfusionMatrixMetrics(threshold=threshold, ignore_index=ignore_index)
    roc_auc_score = ROCAUCScore("binary", ignore_index)
    perf_funcs = [
        WrapDict(conf_metrics, ["Accuracy", "IoU", "Precision", "Recall", "Dice"]),
        WrapDict(roc_auc_score, ["AUC"])
    ]

    # Dict to store performance metrics
    metrics = {}

    pbar = tqdm(
        ds_test,
        desc="Testing",
        unit="imgs",
        dynamic_ncols=True,
    )
    for img, target in pbar:
        img = img.to(device)
        target = target.to(device)

        if args.tta_type == "none":
            scores = model(img.unsqueeze(0))[0]
        else:
            scores = test_util.predict_tta(model, img, transforms, type=args.tta_type)

        # Resize scores to the same size as the target
        original_sz = target.shape[-2:]
        scores = tv_transf_F.resize(scores, original_sz)

        scores = scores.unsqueeze(0)
        target = target.unsqueeze(0)

        for perf_func in perf_funcs:
            # Apply performance metric function
            results = perf_func(scores, target)
            # Iterate over the results and log them
            for name, value in results.items():
                value = value.item()
                if name not in metrics:
                    metrics[name] = [value]
                else:
                    metrics[name].append(value)

    # Create dataframe with metrics
    names = [name.stem for name in ds_test.images]
    metrics_df = pd.DataFrame(metrics, index=names)

    stats_df = metrics_df.describe(percentiles=[])
    stats_df = stats_df.drop("count")

    # Create persistent data
    # Use provided inference_dir_name (default 'inference_results').
    # NOTE: Orquestrador (few_shot_train.py) assume 'inference_results' ao validar.
    # Alterar o nome exige atualizar a lógica de validação/agregação.
    inference_path = Path(run_path) / args.inference_dir_name
    shutil.rmtree(inference_path, ignore_errors=True)
    Path.mkdir(inference_path)

    if args.save_inference_images:
        inference_images_path = inference_path / "inferences"
        inference_images_path.mkdir(parents=True, exist_ok=True)

    for idx, (img, target) in enumerate(pbar):
        img = img.to(device)
        target = target.to(device)
        img_name = Path(ds_test.images[idx]).stem

        if args.tta_type == "none":
            scores = model(img.unsqueeze(0))[0]
        else:
            scores = test_util.predict_tta(model, img, transforms, type=args.tta_type)

        # Resize scores to the same size as the target
        original_sz = target.shape[-2:]
        scores = tv_transf_F.resize(scores, original_sz)

        scores = scores.unsqueeze(0)
        target = target.unsqueeze(0)

        for perf_func in perf_funcs:
            results = perf_func(scores, target)
            for name, value in results.items():
                if hasattr(value, "item"):
                    value = value.item()  # se ainda for tensor
                if name not in metrics:
                    metrics[name] = [value]
                else:
                    metrics[name].append(value)

        # Save inference images
        if args.save_inference_images:
            with torch.no_grad():
                if args.tta_type == "none":
                    output = model(img.unsqueeze(0))[0]
                else:
                    output = test_util.predict_tta(model, img, transforms, type=args.tta_type)

                # Get prediction using argmax and normalize to [0, 255]
                prediction = torch.argmax(output, dim=0)
                prediction = (255 * prediction / (output.shape[0] - 1)).to(torch.uint8)

            # Save the prediction as an image
            img_name = Path(ds_test.images[idx]).stem  # Extract original image name
            inference_image_path = inference_images_path / f"{img_name}.png"
            pil_img = Image.fromarray(prediction.cpu().numpy())  # Convert tensor to PIL image
            pil_img.save(inference_image_path)

    metrics_df.to_csv(inference_path / "metrics.csv", index_label="image")
    stats_df.to_csv(inference_path / "metrics_stats.csv", index_label="statistic")
    if not getattr(args, 'skip_boxplot', False):
        # Import pyplot only if we'll generate the figure (saves import time in headless massive loops)
        ax = metrics_df.boxplot()
        ax.figure.savefig(inference_path / "boxplot.png")

    config_dict = vars(args)
    config_dict["threshold"] = threshold
    args_yaml = yaml.safe_dump(config_dict, default_flow_style=False)
    with open(inference_path / "test_config.yaml", "w") as file:
        file.write(args_yaml)

    return ds_test, metrics_df


def get_args(param_dict: dict | None = None) -> argparse.Namespace:
    """Parse command line arguments or arguments from a string.

    Parameters
    ----------
    param_dict
        A dictionary containing the command line arguments to parse.

    Returns
    -------
    args
        An argparse namespace object containing the parsed arguments
    """

    if param_dict is None:
        sys_argv = None
    else:
        positional_args = ["dataset_path", "dataset_class", "model_class"]
        sys_argv = dict_to_argv(param_dict, positional_args)

    parser = get_parser()

    args = parser.parse_args(sys_argv)

    return args


def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for the `test` function."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--run_path", default="no_name_run", metavar="NAME",
                        help="Path to the run data of an experiment")
    parser.add_argument("--tta_type", type=str, default="none", choices=["none", "logits", "probs"],
                        help="Test-time augmentation type. The logits and probs options set the "
                             "type of values used for TTA averaging.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold to use for binary classification. If -1, uses a threshold "
                             "that maximizes the Dice score on the training dataset.")

    # Dataset parameters
    parser.add_argument("dataset_path", help="Path to the dataset root directory")
    parser.add_argument("dataset_class", help="Name of the dataset class to use")
    parser.add_argument("--resize_size", default=(384, 384), nargs=2, type=int, metavar=("N", "N"),
                        help="Size to resize the images. E.g. --resize_size 128 128")
    parser.add_argument("--dataset_params", nargs="*", default={}, action=ParseKwargs,
                        metavar="par1=v1 par2=v2 par3=v3",
                        help="Additional parameters to pass to the dataset creation function. "
                             "E.g. --dataset_params par1=v1 par2=v2 par3=v3. The additional "
                             "parameters are evaluated as Python code and cannot contain spaces.")
    parser.add_argument("--save_inference_images", action="store_true",
                        help="If set, saves inference images to a directory.")
    parser.add_argument("--inference_dir_name", default="inference_results", metavar="NAME",
                        help="Name of the directory to save inference images.")

    # Model parameters
    parser.add_argument("model_class", help="Name of the trained model")
    parser.add_argument("--model_params", nargs="*", default={}, action=ParseKwargs,
                        metavar="par1=v1 par2=v2 par3=v3",
                        help="Additional parameters to pass to the model creation function. "
                             "E.g. --model_params par1=v1 par2=v2 par3=v3")
    parser.add_argument("--checkpoint_type", default="best", choices=["best", "last"],
                    help="Type of checkpoint to load. Options are 'best' and 'last'.")
    parser.add_argument(
        "--encoder_weights",
        default=None,
        choices=[None, "imagenet"],
        help="Pretrained weights for the encoder backbone (use 'imagenet' for ImageNet pretraining or None for random initialization)."
    )
    parser.add_argument("--skip_checkpoint_loading", action="store_true",
                        help="If set, skips loading model weights from checkpoint and uses random weights or encoder weights if specified.")

    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="Seed for the random number generator")

    # Device and efficiency parameters
    group = parser.add_argument_group("Device and efficiency parameters")
    group.add_argument("--device", default="cuda:0",
                       help='Where to run the test code (e.g. "cpu" or "cuda:0")')
    group.add_argument("--use_amp", action="store_true",
                       help="If automatic mixed precision should be used")
    group.add_argument("--deterministic", action="store_true",
                       help="If deterministic algorithms should be used")
    group.add_argument("--benchmark", action="store_true",
                       help="If cuda benchmark should be used")

    # Inference orchestration helper flags (passed via few_shot_train orchestrator)
    parser.add_argument('--force_headless', action='store_true',
                        help='If set, use a headless Matplotlib backend (Agg).')
    parser.add_argument('--skip_boxplot', action='store_true',
                        help='If set, skip boxplot generation to save time and avoid backend usage.')

    return parser


if __name__ == "__main__":
    test()