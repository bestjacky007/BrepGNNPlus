from datasets.base import BaseDataset
import pathlib
import torch
import dgl

class MFCAD2Dataset(BaseDataset):
    @staticmethod
    def num_classes():
        return 25

    def __init__(
        self, root_dir, split="train", center_and_scale=True, random_rotate=False,
    ):
        """
        Load the MFCAD2 dataset.
        Structure:
        root_dir/
            train/
                *.bin
            val/
                *.bin
            test/
                *.bin
        """
        path = pathlib.Path(root_dir)
        self.path = path
        assert split in ("train", "val", "test")
        
        self.split_path = path.joinpath(split)
        self.random_rotate = random_rotate

        # Get all .bin files in the split directory
        all_files = list(self.split_path.glob("*.bin"))

        # Load graphs
        print(f"Loading {split} data from {self.split_path}...")
        self.load_graphs(all_files, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        # Load the graph using base class method
        # This assumes the .bin file contains the graph with labels (ndata['y'])
        sample = super().load_one_graph(file_path)
        return sample
