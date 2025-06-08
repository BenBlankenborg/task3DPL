import sys
sys.path.insert(0, '../../..')

from trainer_line_ctc import TrainerLineCTC
from models_line_ctc import Decoder
from basic.models import FCN_Encoder
from torch.optim import Adam
from basic.generic_dataset_manager import OCRDataset
import torch.multiprocessing as mp
import torch


def train_and_test(rank, params):
    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    
    # Train the model
    model.train()

    # Load best model
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    # Evaluate on train/valid/test sets
    metrics = ["cer", "wer", "time", "worst_cer"]
    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["train", "valid", "test"]:
            model.predict(f"{dataset_name}-{set_name}", [(dataset_name, set_name)], metrics, output=True)


if __name__ == "__main__":
    dataset_name = "GERMAN"

    params = {
        "dataset_params": {
            "datasets": {
                dataset_name: "/home3/s4895606/task3DLP/Datasets/GERMAN_lines",
            },
            "train": {
                "name": "GERMAN-train",
                "datasets": ["GERMAN"]
            },
            "valid": {
                "GERMAN-valid": {  # Must be a dictionary
                    "name": "GERMAN-valid",
                    "datasets": ["GERMAN"]  # List of dataset keys
                }
            },
            "test": {
                "GERMAN-test": {  # Must be a dictionary
                    "name": "GERMAN-test",
                    "datasets": ["GERMAN"]
                }
            },
            "dataset_class": OCRDataset,
            "config": {
                "width_divisor": 8,
                "height_divisor": 32,
                "padding_value": 0,
                "padding_token": 1000,
                "charset_mode": "CTC",
                "constraints": ["CTC_line"],
                "preprocessings": [
                    {
                        "type": "dpi",
                        "source": 300,
                        "target": 150,
                    },
                    {
                        "type": "to_RGB",
                    },
                ],
                "augmentation": {
                    "dpi": {
                        "proba": 0.2,
                        "min_factor": 0.75,
                        "max_factor": 1.25,
                    },
                    "perspective": {
                        "proba": 0.2,
                        "min_factor": 0,
                        "max_factor": 0.3,
                    },
                    "elastic_distortion": {
                        "proba": 0.2,
                        "max_magnitude": 20,
                        "max_kernel": 3,
                    },
                    "random_transform": {
                        "proba": 0.2,
                        "max_val": 16,
                    },
                    "dilation_erosion": {
                        "proba": 0.2,
                        "min_kernel": 1,
                        "max_kernel": 3,
                        "iterations": 1,
                    },
                    "brightness": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "contrast": {
                        "proba": 0.2,
                        "min_factor": 0.01,
                        "max_factor": 1,
                    },
                    "sign_flipping": {
                        "proba": 0.2,
                    },
                },
            }
        },

        "model_params": {
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder,
            },
            "transfer_learning": None,
            "input_channels": 3,
            "dropout": 0.5,
        },

        "training_params": {
            "output_folder": "german_lines",
            "max_nb_epochs": 5000,
            "max_training_time": 3600 * (24 + 22),
            "load_epoch": "best",
            "interval_save_weights": None,
            "use_ddp": False,
            "use_apex": True,
            "nb_gpu": torch.cuda.device_count(),
            "batch_size": 8,
            "optimizer": {
                "class": Adam,
                "args": {
                    "lr": 0.0001,
                    "amsgrad": True,
                }
            },
            "eval_on_valid": True,
            "eval_on_valid_interval": 2,
            "focus_metric": "cer",
            "expected_metric_value": "low",
            "set_name_focus_metric": f"{dataset_name}-valid",
            "train_metrics": ["loss_ctc", "cer", "wer"],
            "eval_metrics": ["loss_ctc", "cer", "wer"],
            "force_cpu": False,
        },
    }

    if params["training_params"]["use_ddp"] and not params["training_params"]["force_cpu"]:
        mp.spawn(train_and_test, args=(params,), nprocs=params["training_params"]["nb_gpu"])
    else:
        train_and_test(0, params)
