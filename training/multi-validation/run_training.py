from train import MultiTrainer

# The parameters below reproduce those used in the lwnet paper
params1 = {
     "experiment_name": "auto_vess_shape_hiperparam_tuning",
    "run_name": "resnet18_ts:50k_bs-train:200_ep:20_lr:0.01_lr-decay:1.0_wd:0.0_opt:adam_class1:0.35_FP16",
    "validate_every": 1,
    "save_val_imgs": "",
    "val_img_indices": "0 1 2 3",
    "dataset_path": "/media/wesleygalvao/1_TB_LINUX/Datasets/blood_vessels/",
    "dataset_class": "multiple",
    "model_class": "resnet18_unet",
    "num_epochs": 20,
    "validation_metric": "Val loss VessMAP",
    #"maximize_validation_metric": "",
    "bs_train": 200,
    "bs_valid": 16,
    "weight_decay": 0.0,
    "lr": 0.01,
    "lr_decay": 1.0,
    "optimizer": "adam",
    "num_workers": 8,
    "log_wandb": "",
    "wandb_project": "auto_vess_shape_hiperparam_tuning"
}

params2 = {
     "experiment_name": "auto_vess_shape_hiperparam_tuning",
    "run_name": "resnet18_ts:50k_bs-train:200_ep:20_lr:0.01_lr-decay:1.5_wd:0.0_opt:adam_class1:0.35_FP16",
    "validate_every": 1,
    "save_val_imgs": "",
    "val_img_indices": "0 1 2 3",
    "dataset_path": "/media/wesleygalvao/1_TB_LINUX/Datasets/blood_vessels/",
    "dataset_class": "multiple",
    "model_class": "resnet18_unet",
    "num_epochs": 20,
    "validation_metric": "Val loss VessMAP",
    #"maximize_validation_metric": "",
    "bs_train": 200,
    "bs_valid": 16,
    "weight_decay": 0.0,
    "lr": 0.01,
    "lr_decay": 1.5,
    "optimizer": "adam",
    "num_workers": 8,
    "log_wandb": "",
    "wandb_project": "auto_vess_shape_hiperparam_tuning"
}

params3 = {
      "experiment_name": "auto_vess_shape_hiperparam_tuning",
    "run_name": "resnet18_ts:50k_bs-train:200_ep:20_lr:0.01_lr-decay:0.5_wd:0.0_opt:adam_class1:0.35_FP16",
    "validate_every": 1,
    "save_val_imgs": "",
    "val_img_indices": "0 1 2 3",
    "dataset_path": "/media/wesleygalvao/1_TB_LINUX/Datasets/blood_vessels/",
    "dataset_class": "multiple",
    "model_class": "resnet18_unet",
    "num_epochs": 20,
    "validation_metric": "Val loss VessMAP",
    #"maximize_validation_metric": "",
    "bs_train": 200,
    "bs_valid": 16,
    "weight_decay": 0.0,
    "lr": 0.01,
    "lr_decay": 0.5,
    "optimizer": "adam",
    "num_workers": 8,
    "log_wandb": "",
    "wandb_project": "auto_vess_shape_hiperparam_tuning"
}

params4 = {
     "experiment_name": "auto_vess_shape_hiperparam_tuning",
    "run_name": "resnet50_ts:50k_bs-train:100_ep:20_lr:0.001_lr-decay:0.0_wd:0.0_opt:adam_class1:0.35_FP16",
    "validate_every": 1,
    "save_val_imgs": "",
    "val_img_indices": "0 1 2 3",
    "dataset_path": "/media/wesleygalvao/1_TB_LINUX/Datasets/blood_vessels/",
    "dataset_class": "multiple",
    "model_class": "resnet50_unet",
    "num_epochs": 20,
    "validation_metric": "Val loss VessMAP",
    #"maximize_validation_metric": "",
    "bs_train": 100,
    "bs_valid": 8,
    "weight_decay": 0.0,
    "lr": 0.001,
    "lr_decay": 0.0,
    "optimizer": "adam",
    "num_workers": 8,
    "log_wandb": "",
    "wandb_project": "auto_vess_shape_hiperparam_tuning"
}

print("Training Setup 1")
runner1 = MultiTrainer(params1).fit()
print("Training Setup 2")
runner2 = MultiTrainer(params2).fit()
print("Training Setup 3")
runner3 = MultiTrainer(params3).fit()
# print("Training Setup 4")
# runner4 = MultiTrainer(params4).fit()
