import sys
sys.path.append('/mnt/sdc/yangling/MedXchat')
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper_qwenvl import ParseDataset_RG, ParseDataset_VQA, ParseDataset_SD, ParseDataset_VG,RandomDataset
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random


def collate_func(batch):
    elem = batch[0]
    res = {}
    for key, v in elem.items():
        value = [d[key] for d in batch]
        if isinstance(v, str):
             res[key] = value
        elif isinstance(v, torch.Tensor):
             if 'input_ids' in key:
                value = pad_sequence(value, batch_first=True)
             else:
                value = torch.stack(value, 0)
             res[key] = value
        elif isinstance(v, np.ndarray):
             value = torch.tensor(np.stack(value))
             res[key] = value
        elif isinstance(v, int):
             res[key] = torch.tensor(value)
        else:
             print(key)
             print('unkown data type')
    return res


def custom_collate_fn(batch):
    ids = [item["id"] for item in batch]
    conversations = [item["conversations"] for item in batch]
    return {"ids": ids, "conversations": conversations}


class DataModule(LightningDataModule):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.train_dataset = RandomDataset([ParseDataset_RG(args, 'train'), ParseDataset_VQA(args, 'train'), ParseDataset_SD(args, 'train'), ParseDataset_VG(args, 'train')])
        self.dev_dataset = RandomDataset([ParseDataset_RG(args, 'val'), ParseDataset_VQA(args, 'val'), ParseDataset_SD(args, 'val'), ParseDataset_VG(args, 'val')])
        self.test_dataset = RandomDataset([ParseDataset_RG(args, 'test'), ParseDataset_VQA(args, 'test'), ParseDataset_SD(args, 'test'), ParseDataset_VG(args, 'test')])

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """
        # rg_train_dataset, rg_dev_dataset, rg_test_dataset = create_rg_datasets(self.args)
        # vqa_train_dataset, vqa_dev_dataset, vqa_test_dataset = create_vqa_datasets(self.args)
        # sd_train_dataset, sd_dev_dataset, sd_test_dataset = create_sd_datasets(self.args)
        # self.rg_dataset = {
        #     "train": rg_train_dataset, "validation": rg_dev_dataset, "test": rg_test_dataset
        # }
        # self.vqa_dataset = {
        #     "train": vqa_train_dataset, "validation": vqa_dev_dataset, "test": vqa_test_dataset
        # }
        # self.sd_dataset = {
        #     "train": sd_train_dataset, "validation": sd_dev_dataset, "test": sd_test_dataset
        # }
        self.dataset = {
            "train": self.train_dataset, "validation": self.dev_dataset, "test": self.test_dataset
        }


    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, collate_fn=custom_collate_fn)
        return loader


    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=True,
                            num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, collate_fn=custom_collate_fn)
        return loader


    def test_dataloader(self):
        loader = DataLoader(self.dataset["test"], batch_size=self.args.val_batch_size, drop_last=False, pin_memory=False,
                        num_workers=self.args.num_workers, prefetch_factor=self.args.prefetch_factor, collate_fn=custom_collate_fn)
        return loader
    

if __name__ == '__main__':
    from configs.config_qwen import parser
    args = parser.parse_args()
    loader = DataModule(args)
    loader.setup(stage=None)
    train = loader.train_dataloader()

    for data in train:
        print(data)