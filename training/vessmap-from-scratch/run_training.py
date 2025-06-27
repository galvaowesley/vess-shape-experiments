from train import VesselTrainer

# The parameters below reproduce those used in the lwnet paper
params = {
    "experiment_name": "training_on_vessmap_from_scratch",
    "run_name": "resnet50_ts:80_bs-train:8_ep:1000_lr:0.01_lr-decay:1.0_wd:0.0_opt:adam_val-metric:Dice_FP16",
    "validate_every": 5,
    "save_val_imgs": "",
    "val_img_indices": "0 1 2 3",
    "dataset_path": "/home/wesleygalvao/Documents/Datasets/blood_vessels/VessMAP",
    "dataset_class": "vessmap",
    "split_strategy": "file",
    "resize_size": "256 256",
    #'ignore_class_weights': '',
    "model_class": "resnet50_unet",
    "num_epochs": 1000,
    "validation_metric": "Dice",
    "maximize_validation_metric": "",
    "bs_train": 8,
    "bs_valid": 4,
    "weight_decay": 0.0,
    "lr": 0.01,
    "lr_decay": 1.0,
    "optimizer": "adam",
    "num_workers": 8,
    "log_wandb": "",
    "wandb_project": "training_on_vessmap_from_scratch",
}

runner = VesselTrainer(params).fit()
