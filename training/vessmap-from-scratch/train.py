from torchtrainer.train import DefaultTrainer
import torch.nn as nn
import torch
import segmentation_models_pytorch as smp


class VesselTrainer(DefaultTrainer):
    def get_model(self, model_class, weights_strategy, num_classes, num_channels, **model_params):
        """
        Recebe parâmetros do modelo e retorna a instância do modelo.

        Args:
            model_class (str): Nome da classe do modelo.
            weights_strategy (str | None): Estratégia para carregar os pesos.
            num_classes (int): Número de classes de saída.
            num_channels (int): Número de canais de entrada.
            **model_params: Parâmetros adicionais para o modelo.

        Returns:
            torch.nn.Module: Instância do modelo.

        Raises:
            NotImplementedError: Se o model_class não for reconhecido.
        """
        print(f"Carregando modelo '{model_class}' via VessShapeTrainer.get_model()...")

        
        if model_class == "resnet18_unet":
            model = smp.Unet(
                encoder_name="resnet18",
                encoder_weights=None, 
                in_channels=num_channels,
                classes=num_classes,
            )
        elif model_class == "resnet50_unet":
            model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None, 
                in_channels=num_channels,
                classes=num_classes,
            )
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
                state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                model.load_state_dict(state_dict)
                print("Weights loaded successfully.")
            else:
                raise FileNotFoundError(f"The path for weights_strategy is invalid: {weights_strategy}")


        return model

if __name__ == "__main__":
    # This is required to run the script from the command line
    VesselTrainer().fit()