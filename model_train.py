import gc
from pathlib import Path
from typing import Dict

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
from validation import validate_model

cstm_logger = create_logger("train")


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


class ModelTrainer:
    def __init__(self, model: nn.Module,
                 dataloaders_dict: Dict[str, torch.utils.data.DataLoader],
                 model_out_dir: Path,
                 criterion,
                 optimizer: Optimizer,
                 weight_coef_l1: float,
                 weight_coef_l2: float,
                 writer: SummaryWriter,
                 batch_size: int,
                 num_classes: int,
                 num_epochs: int,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.weight_coef_l1 = weight_coef_l1
        self.weight_coef_l2 = weight_coef_l2
        self.writer = writer
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.device = device
        self.model = model.to(self.device)
        cstm_logger.info("Trainer created")

    def train(self, use_group_loss_component: bool = False):
        weight_coef_l1 = torch.as_tensor(self.weight_coef_l1).to(self.device)
        weight_coef_l2 = torch.as_tensor(self.weight_coef_l2).to(self.device)

        # Train the model
        total_step = len(self.dataloader_train)

        l = 0
        elems = []
        last_elem = []
        if use_group_loss_component:
            for param in self.model.parameters():
                if l % 4 == 0:
                    last_elem = []
                last_elem.append(param)
                if l % 4 == 3:
                    elems.append(last_elem)
                l += 1

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

                if use_group_loss_component:
                    for params in elems:
                        weights, bias, norm_coef, norm_bias = params
                        loss += block_regularization_loss_from_weights(weights,
                                                                       bias,
                                                                       norm_coef,
                                                                       norm_bias,
                                                                       weight_coef_l1,
                                                                       weight_coef_l2)
                loss_reg_accum += float(loss.item())
                # Backward and optimize
                # May be zero grad can be deleted?
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                del images, labels, outputs
                clear_cache()

            acc, valid_loss = validate_model(self.model, self.dataloader_valid, self.device, self.criterion)
            acc_train = correct / total

            self.writer.add_scalar("Loss/train", loss_accum, epoch)
            self.writer.add_scalar("Loss_reg/train", loss_reg_accum, epoch)
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)

            self.writer.add_scalar("Acc/train", acc_train, epoch)
            self.writer.add_scalar("Acc/valid", acc, epoch)

            self.writer.add_scalar("Mean_weights", calc_mean_weights(self.model), epoch)

            torch.save(self.model.state_dict(), str(self.model_out_dir / f"ep_{epoch:03d}_acc_{acc:04f}.bin"))

        acc, test_loss = validate_model(self.model, self.dataloader_test, self.device, self.criterion)
        print(f'Accuracy of the network on the {len(self.dataloader_test)} '
              f'test batches: {100 * acc} Loss: {test_loss:04f}')
        torch.save(self.model.state_dict(), str(self.model_out_dir / f"result_acc_{acc:04f}.bin"))

    def __del__(self):
        del self.model
        del self.optimizer
        clear_cache()


if __name__ == "__main__":
    dirs_struct_entity = DirsStruct()
    model_experiment_path, stats_experiment_path = dirs_struct_entity.get_stats__and_model_save_path("module_test")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 456
    cifar10_dataset_creator = Cifar10CSTMDatasetCreator()
    train_valid_dataloaders = cifar10_dataset_creator.create_loaders()
    train_valid_dataloaders["test"] = cifar10_dataset_creator.create_loaders(create_test_dataloader=True)["test"]

    weight_coef_l1 = 1e-10
    weight_coef_l2 = 1e-9
    num_classes = 10
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-8

    model = ResNet(ResidualBlock, [3, 1, 1, 3]).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    writer = SummaryWriter(str(stats_experiment_path), filename_suffix="tsbrd")

    model_trainer = ModelTrainer(
        model, train_valid_dataloaders, model_experiment_path, criterion, optimizer, weight_coef_l1, weight_coef_l2,
        writer, batch_size, num_classes, num_epochs, device
    )

    model_trainer.train()
