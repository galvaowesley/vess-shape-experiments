from pathlib import Path
from torchtrainer.train import DefaultTrainer
from torchtrainer.models.litemedsam.litemedsam import get_model as get_litemedsam_model
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp
from dataset import get_dataset_vessmap_train, get_dataset_drive_train, get_dataset_dca1_train, get_dataset_octa2d_train


class VesselTrainer(DefaultTrainer):
    """
    Specialized trainer for blood vessel segmentation tasks, using the DefaultTrainer structure from the torchtrainer library.
    """
    def get_model(
        self,
        model_class,
        weights_strategy,
        num_classes,
        num_channels,
        freeze_image_encoder=False,
        img_size=(256, 256),
        **model_params
    ):
        """
        Receives model parameters and returns the model instance.

        Args:
            model_class (str): Name of the model class.
            weights_strategy (str | None): Strategy for loading weights.
            encoder_weights (str | None): Encoder weights for initialization.
            num_classes (int): Number of output classes.
            num_channels (int): Number of input channels.
            freeze_image_encoder (bool): If True, freezes the encoder weights (only for SAM models).
            img_size (tuple): Input image size.
            **model_params: Additional model parameters.

        Returns:
            torch.nn.Module: Model instance.

        Raises:
            NotImplementedError: If the model_class is not recognized.
        """
        print(f"Loading model '{model_class}' via VessShapeTrainer.get_model()...")
        
        encoder_weights = getattr(self.args, "encoder_weights", None)

        
        if model_class == "resnet18_unet":
            model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights=encoder_weights,
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
        elif model_class == "litemedsam":
            img_size = getattr(self.args, "resize_size", img_size)
            model = get_litemedsam_model(img_size=img_size, freeze_image_encoder=freeze_image_encoder)
        else:
            raise NotImplementedError(f"Model '{model_class}' not implemented.")

        # If weights_strategy is provided, load the model weights
        # This is useful for transfer learning or fine-tuning
        # If weights_strategy is None, the model will be initialized with random weights
        if weights_strategy is not None:
            weights_path = Path(weights_strategy)
            if weights_path.is_file():
                print(f"Loading weights from {weights_path}...")
                checkpoint = torch.load(weights_path, map_location=self.args.device, weights_only=False)
                 # Check if the checkpoint contains 'model' key
                raw_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                state_dict = {k: v.clone() for k, v in raw_state_dict.items()}
                model.load_state_dict(state_dict)
                print("Weights loaded successfully.")
            else:
                raise FileNotFoundError(f"The path for weights_strategy is invalid: {weights_strategy}")


        return model
    
    def get_dataset(self, dataset_class, dataset_path, split_strategy, resize_size, 
                    augmentation_strategy, **dataset_params):
        channels = getattr(self.args, "channels", "all")
        loss_function = getattr(self.args, "loss_function", "cross_entropy")
        squeeze_target = (loss_function == "cross_entropy")

        if dataset_class == "vessmap_few":
            ds_train, ds_valid, *dataset_props = get_dataset_vessmap_train(
                dataset_path, channels=channels, split_strategy=split_strategy, resize_size=resize_size, squeeze_target=squeeze_target
            )
            class_weights, ignore_index, collate_fn = dataset_props  
        if dataset_class == "drive_few":
            ds_train, ds_valid, class_weights, ignore_index, collate_fn = get_dataset_drive_train(
                dataset_path, split_strategy, resize_size=resize_size, squeeze_target=squeeze_target
            )
        if dataset_class == "dca1_few":
            ds_train, ds_valid, class_weights, ignore_index, collate_fn = get_dataset_dca1_train(
                dataset_path, split_strategy, resize_size=resize_size, squeeze_target=squeeze_target
        ) 
        if dataset_class == "octa2d_few":
            ds_train, ds_valid, class_weights, ignore_index, collate_fn = get_dataset_octa2d_train(
                dataset_path, split_strategy, resize_size=resize_size, squeeze_target=squeeze_target
        ) 
                   
       
        return ds_train, ds_valid, class_weights, ignore_index, collate_fn

if __name__ == "__main__":
    # This is required to run the script from the command line
    VesselTrainer().fit()