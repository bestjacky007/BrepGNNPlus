import pathlib
import string

import torch
from sklearn.model_selection import train_test_split

from datasets.base import BaseDataset


def _get_filenames(root_dir, filelist):
    with open(str(root_dir / f"{filelist}"), "r") as f:
        file_list = [x.strip() for x in f.readlines()]

    files = list(
        x
        for x in root_dir.rglob(f"*.bin")
        if x.stem in file_list
        #if util.valid_font(x) and x.stem in file_list
    )
    return files


CHAR2LABEL = {char: i for (i, char) in enumerate(string.ascii_lowercase)}


def _char_to_label(char):
    return CHAR2LABEL[char.lower()]


class SolidLetters(BaseDataset):
    @staticmethod
    def num_classes():
        return 26

    def __init__(
        self,
        root_dir,
        split="train",
        center_and_scale=True,
        random_rotate=False,
    ):
        """
        Load the SolidLetters dataset with 70/15/15 split
        """
        assert split in ("train", "val", "test")
        path = pathlib.Path(root_dir)

        self.random_rotate = random_rotate

        train_files_raw = _get_filenames(path, filelist="train.txt")
        test_files_raw = _get_filenames(path, filelist="test.txt")
        all_files = train_files_raw + test_files_raw

        all_labels = [_char_to_label(fn.stem[0]) for fn in all_files]

        train_val_files, test_files, train_val_labels, _ = train_test_split(
            all_files, all_labels, test_size=0.15, random_state=42, stratify=all_labels
        )

        if split == "test":
            file_paths = test_files
        else:
            train_files, val_files = train_test_split(
                train_val_files, test_size=0.1765, random_state=42, stratify=train_val_labels
            )
            
            if split == "train":
                file_paths = train_files
            elif split == "val":
                file_paths = val_files

        print(f"Loading {split} data...")
        self.load_graphs(file_paths, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        sample = super().load_one_graph(file_path)
        sample["label"] = torch.tensor([_char_to_label(file_path.stem[0])]).long()
        return sample

    def _collate(self, batch):
        collated = super()._collate(batch)
        collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        return collated
