import gc
import json
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple, Optional

import torch.nn as nn
import torch.optim
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from regularizations import *
from src.dataset_loader import Cifar10CSTMDatasetCreator
from src.dirs_struct import DirsStruct
from src.loggers import create_logger
from src.model import ResNet, ResidualBlock
from src.prunner import Prunner
from validation import validate_model

cstm_logger = create_logger("train")


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


def cut_worst_epoches(experiment_path: Path, last_amount: int):
    epoches_list = list(sorted([(x, float(str(x.name).split("acc_")[-1][:-4])) for x in experiment_path.glob("*.bin")],
                               key=lambda x: x[1]))
    to_delete = epoches_list[:-last_amount]
    for epoch_bin in to_delete:
        os.remove(epoch_bin[0])


class ModelTrainer:
    def __init__(self, model: nn.Module,
                 prunner_obj: Optional[Prunner],
                 dataloaders_dict: Dict[str, torch.utils.data.DataLoader],
                 model_out_dir: Path,
                 criterion,
                 optimizer: Optimizer,
                 pruning_coefficients: Tuple[float],
                 writer: SummaryWriter,
                 batch_size: int,
                 num_classes: int,
                 num_epochs: int,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 loss_up_period: int = 3,
                 cut_worst_amount: int = 10
                 ):
        cstm_logger.info("Creating trainer")
        self.model = model
        if not ("train" in dataloaders_dict and
                "valid" in dataloaders_dict and
                "test" in dataloaders_dict):
            cstm_logger.error(f"Trainer d'nt have all loader. Have only: {dataloaders_dict.keys()}")
            exit(1)
        self.dataloader_train = dataloaders_dict["train"]
        self.dataloader_valid = dataloaders_dict["valid"]
        self.dataloader_test = dataloaders_dict["test"]
        self.model_out_dir = model_out_dir
        self.criterion = criterion
        self.optimizer = optimizer
        self.pruning_coefficients = pruning_coefficients
        self.writer = writer
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.device = device
        self.model = model.to(self.device)
        self.loss_up_period = loss_up_period
        self.cut_worst_amount = cut_worst_amount
        self.prunner_obj = prunner_obj
        cstm_logger.info("Trainer created")

    def train(self):

        # Train the model
        total_step = len(self.dataloader_train)

        valid_last_losses = []
        train_last_losses = []
        for epoch in range(self.num_epochs):
            loss_accum = 0.0
            loss_reg_accum = 0.0
            total = 0
            correct = 0
            for i, (images, labels) in enumerate(tqdm(self.dataloader_train, desc="training", total=total_step)):
                # Move tensors to the configured device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                loss_accum += float(loss.item())

                if prunner_obj:
                    prunner_obj.prune(loss)
                loss_reg_accum += float(loss.item())
                # Backward and optimize
                # May be zero grad can be deleted?
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del images, labels, outputs
                clear_cache()

            acc_train = correct / total
            acc, valid_loss = validate_model(self.model, self.dataloader_valid, self.device, self.criterion)

            train_last_losses.append(loss_accum)
            valid_last_losses.append(valid_loss)

            self.writer.add_scalar("Loss/train", loss_accum, epoch)
            self.writer.add_scalar("Loss_reg/train", loss_reg_accum, epoch)
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)

            self.writer.add_scalar("Acc/train", acc_train, epoch)
            self.writer.add_scalar("Acc/valid", acc, epoch)

            self.writer.add_scalar("Mean_weights", calc_mean_weights(self.model), epoch)

            torch.save(self.model.state_dict(), str(self.model_out_dir / f"ep_{epoch:03d}_acc_{acc:04f}.bin"))

            cut_worst_epoches(self.model_out_dir, self.cut_worst_amount)

            if epoch > 0 and min(train_last_losses[-self.loss_up_period - 1:-1]) >= train_last_losses[-1] and max(
                    valid_last_losses[-self.loss_up_period - 1:-1]) < valid_last_losses[-1]:
                print(f"Valid loss stop decreasing for {self.loss_up_period} epoches. Stop training")
                break

    def __del__(self):
        del self.model
        del self.optimizer
        clear_cache()


dataset_list = [("cifar10", Cifar10CSTMDatasetCreator)]

filter_regularization_2coefs_losses_list = [
    ('weights', filter_regularization_loss_from_weights)
]

filter_regularization_1coefs_losses_list = [
    ('entropy', filter_regularization_loss_from_entropy),
    ('entropy-inv', filter_regularization_loss_from_entropy_inv),
    ('radem', filter_regularization_loss_from_rademacher),
    ('radem-inv', filter_regularization_loss_from_rademacher_inv)
]

coefs2 = [
    (1e-10, 1e-9), (1e-9, 1e-8), (1e-8, 1e-7), (1e-7, 1e-6), (1e-6, 1e-5), (1e-5, 1e-4)
]
coefs1 = [
    (1e-10,), (1e-9,), (1e-8,), (1e-7,), (1e-6,), (1e-5,), (1e-4,)
]

init_weights = [
    None,
    Path("/media/kirrog/data/data/fqwb_data/models/2024_12_07__13_34___cifar10/ep_047_acc_0.846400.bin")  # V1
]

experiments_list = []
for name, dataset_class_ in dataset_list:
    for init_path in init_weights:
        for func_name, regs_2coefs in filter_regularization_2coefs_losses_list:
            for coefs_2_instance in coefs2:
                experiment_name = (f"{name}__"
                                   f"{'v1' if init_path else 'none'}__"
                                   f"{func_name}__"
                                   f"{'_'.join([str(x) for x in coefs_2_instance])}")
                experiment = [dataset_class_, experiment_name,
                              regs_2coefs, coefs_2_instance, 10, 50,
                              1e-3,
                              1e-8, True,
                              init_path]
                experiments_list.append(experiment)
        for func_name, regs_1coefs in filter_regularization_1coefs_losses_list:
            for coefs_1_instance in coefs1:
                experiment_name = (f"{name}__"
                                   f"{'v1' if init_path else 'none'}__"
                                   f"{func_name}__"
                                   f"{'_'.join([str(x) for x in coefs_1_instance])}")
                experiment = [dataset_class_, experiment_name,
                              regs_1coefs, coefs_1_instance, 10, 50,
                              1e-3,
                              1e-8, True,
                              init_path]
                experiments_list.append(experiment)
experiments_list = experiments_list[20:]
print(f"Formed: {len(experiments_list)} experiments!")

if __name__ == "__main__":
    pprint(experiments_list)
    for (DatasetCreatorClass,
         experiment_name,
         prunning_func,
         pruning_coefficients,
         num_classes,
         num_epochs,
         learning_rate,
         weight_decay,
         use_group_loss,
         initialisation_path) in experiments_list:
        dirs_struct_entity = DirsStruct()
        model_experiment_path, stats_experiment_path = dirs_struct_entity.get_stats__and_model_save_path(
            experiment_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = 456
        cifar10_dataset_creator = DatasetCreatorClass()
        train_valid_dataloaders = cifar10_dataset_creator.create_loaders()
        train_valid_dataloaders["test"] = cifar10_dataset_creator.create_loaders(create_test_dataloader=True)["test"]

        model = ResNet(ResidualBlock, [3, 1, 1, 3]).to(device)
        print(f"Loading model state from: {initialisation_path}")
        if initialisation_path and initialisation_path.exists():
            model.load_state_dict(torch.load(initialisation_path, weights_only=True))

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if use_group_loss:
            prunner_obj = Prunner(model, pruning_coefficients, device, prunning_func)
        else:
            prunner_obj = None
        print(f"Using pruner: {prunner_obj}")

        writer = SummaryWriter(str(stats_experiment_path), filename_suffix="tsbrd")

        model_trainer = ModelTrainer(
            model,
            prunner_obj,
            train_valid_dataloaders,
            model_experiment_path,
            criterion,
            optimizer,
            pruning_coefficients,
            writer,
            batch_size,
            num_classes,
            num_epochs,
            device,
            5
        )

        model_trainer.train()
