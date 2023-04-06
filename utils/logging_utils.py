import wandb
import csv
import os
from typing import Dict
from collections import deque
from datetime import datetime


class RunningAverage:
    def __init__(self, size):
        self.size = size
        self.q = deque()
        self.sum = 0

    def add(self, val):
        self.q.append(val)
        self.sum += val
        if len(self.q) > self.size:
            self.sum -= self.q.popleft()

    def mean(self):
        # Avoid divide by 0
        return self.sum / max(len(self.q), 1)


def timestamp():
    return datetime.now().strftime("%B %d, %H:%M:%S")


def wandb_init(config, group_keys, **kwargs) -> None:
    wandb.init(
        project=config["project_name"],
        group="_".join(
            [f"{key}={val}" for key, val in config.items() if key in group_keys]
        ),
        config=config,
        **kwargs,
    )


# TODO: Update for multiple envs and/or asymmetric evaluation
class CSVLogger:
    """Logger to write results to a CSV. The log function matches that of Weights and Biases.

    Args:
        path: path for the csv results file
    """

    def __init__(self, path: str):
        self.results_path = path + "_results.csv"
        self.losses_path = path + "_losses.csv"
        # If we have a checkpoint, we don't want to overwrite
        if not os.path.exists(self.results_path):
            with open(self.results_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Step",
                        "Success Rate",
                        "Return",
                        "Episode Length",
                        "Hours",
                        "Mean Success Rate",
                        "Mean Return",
                        "Mean Episode Length",
                    ]
                )
        if not os.path.exists(self.losses_path):
            with open(self.losses_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Step",
                        "TD Error",
                        "Grad Norm",
                        "Max Q Value",
                        "Mean Q Value",
                        "Min Q Value",
                        "Max Target Value",
                        "Mean Target Value",
                        "Min Target Value",
                    ]
                )

    def log(self, results: Dict[str, str], step: int):
        with open(self.results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    step,
                    results["results/Success_Rate"],
                    results["results/Return"],
                    results["results/Episode_Length"],
                    results["results/Hours"],
                    results["results/Mean_Success_Rate"],
                    results["results/Mean_Return"],
                    results["results/Mean_Episode_Length"],
                ]
            )
        with open(self.losses_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    step,
                    results["losses/TD_Error"],
                    results["losses/Grad_Norm"],
                    results["losses/Max_Q_Value"],
                    results["losses/Mean_Q_Value"],
                    results["losses/Min_Q_Value"],
                    results["losses/Max_Target_Value"],
                    results["losses/Mean_Target_Value"],
                    results["losses/Min_Target_Value"],
                ]
            )


def get_logger(policy_path: str, args, wandb_kwargs):
    if args.disable_wandb:
        logger = CSVLogger(policy_path)
    else:
        wandb_init(
            vars(args),
            [
                "model",
                "obs_embed",
                "a_embed",
                "in_embed",
                "context",
                "layers",
                # "eval_context",
                "bag_size",
                # "eval_bag_size",
                # "gate",
                # "identity",
                "history",
                "pos",
            ],
            **wandb_kwargs,
        )
        logger = wandb
    return logger
