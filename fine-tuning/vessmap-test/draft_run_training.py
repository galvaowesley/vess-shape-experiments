from train import VesselTrainer
from torchtrainer.util.train_util import dict_to_argv

# The parameters below reproduce those used in the lwnet paper
weights_path = "/home/wesleygalvao/Documents/vessel-shape-experiments/training/multi-validation/experiments/training_on_auto_vess-shape_multiple_validation/resnet50_ts:50k_bs-train:96_ep:3000_lr:0.001_lr-decay:0.0_wd:0.0001_opt:adam_class1:0.10_FP16/checkpoint.pt"
params = {
    "experiment_name": "finetuning_on_vessmap_test",
    "run_name": "resnet50_ts:80_bs-train:8_ep:1000_lr:0.0001_lr-decay:1.0_wd:0.0_opt:adam_val-metric:Dice_FP16",
    "weights_strategy": weights_path,
    "validate_every": 5,
    "save_val_imgs": "",
    "val_img_indices": "0 1 2 3",
    "dataset_path": "/home/wesleygalvao/Documents/Datasets/blood_vessels/VessMAP",
    "dataset_class": "vessmap_few",
    "split_strategy": "16689,11411",
    "resize_size": "256 256",
    #'ignore_class_weights': '',
    "model_class": "resnet50_unet",
    "num_epochs": 1000,
    "validation_metric": "Dice",
    "maximize_validation_metric": "",
    "bs_train": 8,
    "bs_valid": 4,
    "weight_decay": 0.0,
    "lr": 0.0001,
    "lr_decay": 1.0,
    "optimizer": "adam",
    "num_workers": 8,
    "log_wandb": "",
    "wandb_project": "finetuning_on_vessmap_test",
}



for num_samples in range(1, 20):
    random.seed(num_samples)
    for run_number in range(10):
        
        params["split_strategy"] = ",".join(random.sample(names, num_samples))

        for rep_idx in range(5):
            params["run_name"] = f"r_{run_number}_s_{rep_idx}"
            params["seed"] = rep_idx
        
            commandline = ' '.join(dict_to_argv(params, ["dataset_path", "dataset_class", "model_class"]))
            !python train.py {commandline}

runner = VesselTrainer(params).fit()