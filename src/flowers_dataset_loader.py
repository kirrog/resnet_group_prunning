from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Dict, Tuple

import numpy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.loggers import create_logger

logger = create_logger(__name__)


# logger.setLevel(logging.DEBUG)

class CustomImageDataset(Dataset):
    def __init__(self, img_file, labels_file, transform=None):
        self.annotations = numpy.load(labels_file)
        self.img_data = numpy.load(img_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise Exception("Not implemented")

        image = Image.fromarray(self.img_data[idx])
        class_id = self.annotations[idx]

        if self.transform:
            image = self.transform(image)

        return image, class_id


class FlowersCSTMDatasetCreator:
    def __init__(self, data_dir: Path = Path("/media/kirrog/data/data/fqwb_data/data/flowers/prepared"),
                 batch_size: int = 1024,
                 random_seed: int = 42,
                 valid_size: float = 0.1,
                 shuffle: bool = True,
                 num_of_workers: int = 24,
                 image_size: Tuple[int, int] = (32, 32)):
        if not (data_dir.exists() and data_dir.is_dir()):
            logger.warning(f"Datadir looks wrong: "
                           f"exists: {data_dir.exists()} "
                           f"is_dir: {data_dir.is_dir()} "
                           f"text: {data_dir}")
        else:
            logger.info("Datapath is existed dir")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.valid_size = valid_size
        self.shuffle = shuffle
        self.num_of_workers = num_of_workers
        self.image_size = image_size
        logger.info("FlowersDataCreator init")

    def create_loaders(self, create_test_dataloader: bool = False) -> Dict[str, torch.utils.data.DataLoader]:
        logger.info(f"Create dataloader: is_test:{create_test_dataloader}")
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )

        # define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        if create_test_dataloader:
            dataset = CustomImageDataset(
                self.data_dir / "test_data.npy",
                self.data_dir / "test_labels.npy",
                transform=transform,
            )

            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_of_workers  # , pin_memory=True
            )

            return {"test": data_loader}

        # load the dataset
        train_dataset = CustomImageDataset(
            self.data_dir / "train_data.npy",
            self.data_dir / "train_labels.npy",
            transform=train_transform,
        )

        valid_dataset = CustomImageDataset(
            self.data_dir / "valid_data.npy",
            self.data_dir / "valid_labels.npy",
            transform=train_transform,
        )

        train_indices = list(range(len(train_dataset)))
        valid_indices = list(range(len(valid_dataset)))
        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(train_indices)
            np.random.shuffle(valid_indices)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_of_workers)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_of_workers)

        return {"train": train_loader, "valid": valid_loader}


def converting_archive2format():
    new_size = (32, 32)
    train_part, valid_part, test_part = 0.8, 0.1, 0.1
    p = Path("/media/kirrog/data/data/fqwb_data/data/flowers")
    images_list = list(p.glob("archive/flower_photos/*/*.jpg"))
    labels_ids = dict()
    train_labels_ids_stats = defaultdict(int)
    max_width = 0
    max_height = 0
    output_path = p / "prepared"
    print(f"Save to: {output_path}")
    labels_id2img_list = defaultdict(list)
    for i, img_path in enumerate(tqdm(images_list, desc="Reading")):
        try:
            label_name = str(img_path.parent.name)
            if label_name in labels_ids:
                label_id = labels_ids[label_name]
            else:
                label_id = len(labels_ids)
                labels_ids[label_name] = label_id
            img = Image.open(img_path)
            width_, height_ = img.size
            max_width = max(max_width, width_)
            max_height = max(max_height, height_)
            resized_img = img.resize(new_size)
            resized_img_np = numpy.array(resized_img)
            labels_id2img_list[label_id].append(resized_img_np)
            train_labels_ids_stats[label_id] += 1
        except:
            pass
    pprint(train_labels_ids_stats)
    print(len(train_labels_ids_stats))
    print((max_width, max_height))
    train_img_list = []
    valid_img_list = []
    test_img_list = []
    train_label_ids_list = []
    valid_label_ids_list = []
    test_label_ids_list = []
    for label_id, img_list in labels_id2img_list.items():
        train_valid_size = int(len(img_list) * train_part)
        valid_test_size = train_valid_size + int(len(img_list) * valid_part)
        l = img_list[:train_valid_size]
        train_img_list.extend(l)
        train_label_ids_list.extend([label_id for x in range(len(l))])
        l = img_list[train_valid_size:valid_test_size]
        valid_img_list.extend(l)
        valid_label_ids_list.extend([label_id for x in range(len(l))])
        l = img_list[valid_test_size:]
        test_img_list.extend(l)
        test_label_ids_list.extend([label_id for x in range(len(l))])

    numpy.save(output_path / "train_data.npy", numpy.stack(train_img_list))
    numpy.save(output_path / "train_labels.npy", numpy.array(train_label_ids_list))
    numpy.save(output_path / "valid_data.npy", numpy.stack(valid_img_list))
    numpy.save(output_path / "valid_labels.npy", numpy.array(valid_label_ids_list))
    numpy.save(output_path / "test_data.npy", numpy.stack(test_img_list))
    numpy.save(output_path / "test_labels.npy", numpy.array(test_label_ids_list))


if __name__ == "__main__":
    converting_archive2format()
    # flowers_dataset_creator = FlowersCSTMDatasetCreator()
    # test_dataloader = flowers_dataset_creator.create_loaders(create_test_dataloader=True)["test"]
    # train_valid_dataloaders = flowers_dataset_creator.create_loaders()
    # train_dataloader = train_valid_dataloaders["train"]
    # valid_dataloader = train_valid_dataloaders["valid"]
    # for case in train_dataloader:
    #     print()
