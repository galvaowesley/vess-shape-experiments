from train import MultiTrainer

params1 = {
     "experiment_name": "training_on_auto_vess-shape_multiple_validation",
    "run_name": "resnet18_ts:50k_bs-train:192_ep:3000_lr:0.001_lr-decay:0.0_wd:0.0001_opt:adam_class1:0.10_FP16",
    "validate_every": 5,
    "save_val_imgs": "",
    "val_img_indices": "0 1 2 3 4",
    "dataset_path": "/home/wesleygalvao/Documents/Datasets/blood_vessels/",
    "dataset_class": "multiple",
    "model_class": "resnet18_unet",
    "num_epochs": 3000,
    "validation_metric": "Val loss VessMAP",
    #"maximize_validation_metric": "",
    "bs_train": 192,
    "bs_valid": 8,
    "weight_decay": 0.0001,
    "lr": 0.001,
    "lr_decay": 0.0,
    "optimizer": "adam",
    "num_workers": 12,
    "log_wandb": "",
    "wandb_project": "training_on_auto_vess-shape_multiple_validation",
}

print("Training Setup 1")
runner1 = MultiTrainer(params1).fit()

# runner4 = MultiTrainer(params4).fit()
