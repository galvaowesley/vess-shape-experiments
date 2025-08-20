import operator
from datetime import datetime
from pathlib import Path

import torch
import torch.utils
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp  # necessário para os modelos customizados
from torchtrainer.metrics.confusion_metrics import ConfusionMatrixMetrics
from torchtrainer.train import DefaultModuleRunner, DefaultTrainer
from torchtrainer.util.profiling import Profiler
from torchtrainer.util.train_util import (
    Logger,
    LoggerPlotter,
    WrapDict,
    seed_all,
    seed_worker,
    to_csv_nan,
)
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    has_wandb = False
else:
    has_wandb = True


class MultiModuleRunner(DefaultModuleRunner):
    @torch.no_grad()
    def validate_one_epoch(self, epoch: int):
        self.model.eval()
        profiler = self.profiler

        if epoch == 1:
            profiler.start("validation")

        for ds_name, dl_valid in self.dl_valid.items():
            dl_iter = iter(dl_valid)

            pbar = tqdm(
                range(len(dl_valid)),
                desc="Validating",
                leave=False,
                unit="batchs",
                dynamic_ncols=True,
                colour="green",
                disable=self.args.disable_tqdm,
            )

            for batch_idx in pbar:
                with profiler.section(f"data_{batch_idx}"):
                    imgs, targets = next(dl_iter)
                # print(f"Batch {batch_idx} - {ds_name}")

                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                with profiler.section(f"forward_{batch_idx}"):
                    scores = self.model(imgs)
                    loss = self.loss_func(scores, targets)

                with profiler.section(f"metrics_{batch_idx}"):
                    self.logger.log(epoch, batch_idx, f"Val loss {ds_name}", loss, imgs.shape[0])
                    for perf_func in self.perf_funcs:
                        results = perf_func(scores, targets)
                        for name, value in results.items():
                            self.logger.log(epoch, batch_idx, f"{name} {ds_name}", value, imgs.shape[0])
                profiler.step()


class MultiTrainer(DefaultTrainer):
    def __init__(self, param_dict: dict | None = None):
        args = self.get_args(param_dict)
        seed_all(args.seed)
        self.args = args
        self.module_runner = MultiModuleRunner()
        print("Setting up the experiment...")
        self.setup_experiment()
        self.setup_dataset()
        self.setup_model()
        self.setup_training()
        print("Done setting up.")

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
                encoder_weights=None,  # Ajuste conforme necessário
                in_channels=num_channels,
                classes=num_classes,
            )
            
        elif model_class == "resnet50_unet":
            model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights=None,  # Ajuste conforme necessário
                in_channels=num_channels,
                classes=num_classes,
            )
        else:
            raise NotImplementedError(f"Model '{model_class}' not implemented.")

        if weights_strategy is not None:
            weights_path = Path(weights_strategy)
            if weights_path.is_file():
                print(f"Carregando pesos de {weights_path}...")
                checkpoint = torch.load(weights_path, map_location=self.args.device)
                state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                model.load_state_dict(state_dict)
                print(f"Pesos carregados de {weights_path}.")
            else:
                raise FileNotFoundError(f"The path for weights_strategy is invalid: {weights_strategy}")

        return model

    def setup_dataset(self):
        """Setup the dataset and related elements."""
        args = self.args
        dataset_path = args.dataset_path
        seed_all(args.seed)

        from dataset import get_datasets
        ds_train_vessshape, ds_valids, class_weights, ignore_index = get_datasets(dataset_path)

        if args.loss_function == "cross_entropy":
            loss_func = nn.CrossEntropyLoss(
                torch.tensor(class_weights, device=args.device), 
                ignore_index=ignore_index
            )
        else:
            raise ValueError(f"Loss function {args.loss_function} not recognized")

        conf_metrics = ConfusionMatrixMetrics(ignore_index=ignore_index)
        perf_funcs = [
            WrapDict(conf_metrics, ["Accuracy", "IoU", "Precision", "Recall", "Dice"]),
        ]

        logger = Logger()
        logger_plotter = LoggerPlotter([
            {
                "names": ["Train loss", "Val loss VessShape", "Val loss DRIVE", "Val loss VessMAP"],
                "y_max": 1.
            },
            {
                "names": ["Dice VessShape", "Dice DRIVE", "Dice VessMAP"],
                "y_max": 1.
            }
        ])

        num_classes = 2
        num_channels = 1
        self.module_runner.add_dataset_elements(
            ds_train_vessshape, ds_valids, num_classes, num_channels,
            None, loss_func, perf_funcs, logger, logger_plotter
        )

    def setup_training(self):
        """Setup the training elements: dataloaders, optimizer, scheduler and logger."""
        args = self.args
        num_epochs = args.num_epochs
        optimizer = args.optimizer
        momentum = args.momentum
        module_runner = self.module_runner

        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = args.benchmark
        torch.set_float32_matmul_precision("high")

        num_workers = args.num_workers
        device = args.device

        dl_train = DataLoader(
            module_runner.ds_train,
            batch_size=args.bs_train,
            shuffle=True,
            collate_fn=module_runner.collate_fn,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            pin_memory="cuda" in device,
        )

        dl_valids = {}
        for name, ds in module_runner.ds_valid.items():
            dl_valids[name] = DataLoader(
                ds,
                batch_size=args.bs_valid,
                shuffle=False,
                collate_fn=module_runner.collate_fn,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                pin_memory="cuda" in device,
            )

        model = module_runner.model
        model.to(device)

        try:
            dl = next(iter(dl_valids.values()))
            imgs, targets = next(iter(dl))
        except Exception as e:
            print("The following problem was detected on the validation dataloader:")
            raise e

        try:
            with torch.no_grad():
                scores = model(imgs.to(device))
        except Exception as e:
            print("The following error happened when applying the model to the validation batch:")
            raise e

        try:
            for perf_func in module_runner.perf_funcs:
                perf_func(scores, targets.to(device))
        except Exception as e:
            print("The following error happened when applying the performance functions:")
            raise e

        if optimizer == "sgd":
            optim = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=momentum
            )
        elif optimizer == "adam":
            optim = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(momentum, 0.999)
            )
        elif optimizer == "adamw":
            optim = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(momentum, 0.999)
            )
        
        # Reset _epoch_sample_count
        # dl.dataset.ds.reset()
        

        scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs, args.lr_decay)
        scaler = torch.GradScaler(device=device, enabled=args.use_amp)

        if args.profile:
            args.num_epochs = 2
        profiler = Profiler(
            num_steps=args.profile_batches,
            include_cuda="cuda" in args.device,
            trace_path=self.run_path,
            record_shapes=args.profile_verbosity > 0,
            with_stack=args.profile_verbosity > 1,
            enabled=args.profile,
        )

        self.module_runner.add_training_elements(dl_train, dl_valids, optim, scheduler, scaler, profiler, device)

    def fit(self):
        """Start the training loop."""
        args = self.args
        module_runner = self.module_runner
        logger = module_runner.logger
        logger_plotter = module_runner.logger_plotter
        run_path = self.run_path

        seed_all(args.seed)
        val_metric_name = args.validation_metric
        maximize = args.maximize_validation_metric
        compare = operator.gt if maximize else operator.lt
        best_val_metric = -torch.inf if maximize else torch.inf

        epochs_without_improvement = 0
        print("Training has started")
        pbar = tqdm(
            range(args.num_epochs),
            desc="Epochs",
            leave=True,
            unit="epochs",
            dynamic_ncols=True,
            colour="blue",
            disable=args.disable_tqdm,
        )
        try:
            for epoch in pbar:
                module_runner.train_one_epoch(epoch)
                validate = epoch == 0 or epoch == args.num_epochs - 1 or epoch % args.validate_every == 0

                if validate:
                    module_runner.validate_one_epoch(epoch)

                logger.end_epoch()
                logger_data = logger.get_data()
                last_metrics = logger_data.iloc[-1]

                tqdm_metrics = ["Train loss"]
                if validate:
                    tqdm_metrics += [f"Val loss VessShape", val_metric_name]
                pbar.set_postfix(last_metrics[tqdm_metrics].to_dict())

                to_csv_nan(logger_data, run_path / "log.csv")
                logger_plotter.get_plot(logger).savefig(run_path / "plots.png")

                if args.log_wandb:
                    wandb.log(last_metrics.to_dict())

                checkpoint = module_runner.state_dict()
                torch.save(checkpoint, run_path / "checkpoint.pt")
                if args.copy_model_every and epoch % args.copy_model_every == 0:
                    torch.save(checkpoint, run_path / "models" / f"checkpoint_{epoch}.pt")

                if validate:
                    if args.save_val_imgs:
                        predict_and_save_val_imgs(
                            module_runner, epoch, args.val_img_indices, run_path
                        )
                    val_metric = last_metrics[val_metric_name]
                    if compare(val_metric, best_val_metric):
                        torch.save(checkpoint, run_path / "best_model.pt")
                        best_val_metric = val_metric
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if args.patience is not None and epochs_without_improvement > args.patience:
                            break

        except KeyboardInterrupt:
            pass

        if args.log_wandb:
            wandb.finish()
        print("Training has finished")

        config_dict = vars(args)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        config_dict["timestamp_end"] = timestamp
        args_yaml = yaml.safe_dump(config_dict, default_flow_style=False)
        with open(run_path / "config.yaml", "w") as file:
            file.write(args_yaml)

        return module_runner


def predict_and_save_val_imgs(runner, epoch, img_indices, run_path):
    """Apply model and save validation images for a given epoch."""
    for name, ds in runner.ds_valid.items():
        for img_idx in img_indices:
            img, _ = ds[img_idx]
            output = runner.predict(img.unsqueeze(0))
            prediction = torch.argmax(output, dim=1)
            prediction = 255 * prediction / (runner.num_classes - 1)
            pil_img = Image.fromarray(prediction.squeeze().to(torch.uint8).numpy())
            pil_img.save(run_path / "images" / f"image_{img_idx}" / f"{name}_epoch_{epoch}.png")
    
    # runner.ds_valid['VessShape'].ds.reset()


if __name__ == "__main__":
    MultiTrainer().fit()
