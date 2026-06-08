import os
import sys
from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List, Optional

import optuna
import torch
from optuna.trial import TrialState

from tabularbench.dataloaders import get_custom_dataloader
from tabularbench.datasets.dataset_factory import load_dataset, get_dataset
from tabularbench.metrics.compute import compute_metric
from tabularbench.metrics.metric_factory import create_metric
from tabularbench.models.model_factory import load_model
from tabularbench.models.tab_scaler import TabScaler

# Torch config to avoid crash on HPC
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class TrainParams:
    subset: int = 0
    train_batch_size: int = 1024
    val_batch_size: int = 2048
    epochs: int = 0
    verbose: int = 0
    device: str = "cpu"
    seed: int = 42


def train_save_model(
    dataset_name: str = "lcld_v2_iid",
    model_name: str = "tabtransformer",
    dataloader: str = None,
    train_params: Optional[TrainParams] = None,
    use_optuna: bool = False, # Set this to True to run optimization
    ) -> None:

    if dataloader is None:
        dataloader = "default"
    if train_params is None:
        train_params = TrainParams()

    dataset = get_dataset(dataset_name)
    metadata = dataset.get_metadata(only_x=True)
    x, y = dataset.get_x_y()
    splits = dataset.get_splits()
    
    # Base model arguments
    model_args = {
        "x_metadata": metadata,
        "objective": "classification",
        "use_gpu": True,
        "batch_size": train_params.train_batch_size,
        "num_classes": 2,
        "early_stopping_rounds": train_params.epochs,
        "val_batch_size": train_params.val_batch_size,
        "class_weight": "balanced",
        "custom_dataloader": dataloader,
        "epochs": train_params.epochs,
        "dataset": dataset_name,
        "model_name": model_name,
        "num_splits": 5,
        "seed": train_params.seed,
        "shuffle": True,
        "metrics": ["auc"],
        "learning_rate": 0.001,
        "n_layers": 2,
        "hidden_dim": 10,
    }
    if model_name == "torchrln":
        model_args["weight_decay"] = 0

    # Handle Optimization
    best_params = {}
    if use_optuna:
        # Path setup
        try:
            SCRIPT_DIR = Path(__file__).resolve().parent
        except NameError:
            SCRIPT_DIR = Path(os.getcwd()).resolve()
            
        base_dir = SCRIPT_DIR / "data" / "model_parameters" / dataset_name.replace("/", "-")
        base_dir.mkdir(parents=True, exist_ok=True)
        db_path = base_dir / f"{model_name}_{dataset_name}.db"
        storage_name = f"sqlite:///{db_path.as_posix()}"

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_{dataset_name}",
            storage=storage_name,
            load_if_exists=True,
        )
        
        if len(study.trials) > 0:
            best_params = study.best_trial.params
            print(f"Loaded best params: {best_params}")
        else:
            print("Optuna requested but no trials found in DB.")
            # Decide here if you want to raise an error or continue with defaults

    # Setup model
    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(torch.tensor(x.values, dtype=torch.float32), x_type=metadata["type"])

    # Merge base args with optimized params
    final_model_args = {**model_args, **best_params, "force_device": train_params.device}
    # final_model_args.pop("x_metadata", None) # Clean up
    
    model_class = load_model(model_name)
    model = model_class(**final_model_args, scaler=scaler)

    # Training
    x_train = x.iloc[splits["train"]].values
    y_train = y[splits["train"]]
    if train_params.subset > 0:
        x_train, y_train = x_train[:train_params.subset], y_train[:train_params.subset]

    custom_train_dataloader = get_custom_dataloader(
        dataloader, dataset, model, scaler, {},
        verbose=train_params.verbose, x=x_train, y=y_train,
        train=True, batch_size=train_params.train_batch_size
    )
    
    model.fit(x_train, y_train, x.iloc[splits["test"]], y[splits["test"]], 
              custom_train_dataloader=custom_train_dataloader)

    # Save
    save_path = Path("../data/models") / dataset_name / f"{model_name}_{dataset_name}_{dataloader}.model"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_params = TrainParams(
        subset=0,
        train_batch_size=1024,
        val_batch_size=2048,
        verbose=0,
        epochs=100,
        device="cpu",
    )

    # for dataset in ["ctu_13_neris", "url", "lcld_v2_iid", "malware", "wids"]:
    for dataset in ["url"]:

        # for models in ["tabtransformer", "torchrln", "stg", "tabnet", "vime"]:
        models_list = ["stg", "tabnet", "tabtransformer", "torchrln", "vime",]# "mlp"]
        models_list = ["torchrln", "vime",]# "mlp"]
        for models in models_list:

            for dataloader in [
                "default",
                # "madry",
                # "cutmix",
                # "cutmix_madry",
                # "ctgan",
                # "ctgan_madry",
                # "goggle",
                # "goggle_madry",
                # "tablegan",
                # "tablegan_madry",
                # "tvae",
                # "tvae_madry",
                # "wgan",
                # "wgan_madry",
            ]:

                train_save_model(
                    dataset_name=dataset,
                    model_name=models,
                    dataloader=dataloader,
                    train_params=train_params,
                )
