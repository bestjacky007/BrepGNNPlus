import pathlib
import torch
import dgl
from datasets.base import BaseDataset

def _get_filenames(root_dir, filelist_name):
    filelist_path = root_dir / filelist_name
    if not filelist_path.exists():
        raise FileNotFoundError(f"File list not found: {filelist_path}")
        
    with open(str(filelist_path), "r") as f:
        file_stems = set(x.strip() for x in f.readlines() if x.strip())
    
    files = []
    for stem in file_stems:
        p = root_dir / f"{stem}.bin"
        if p.exists():
            files.append(p)
    files.sort()
    return files

class CustomBinDataset(BaseDataset):
    def __init__(self, root_dir, split="train", center_and_scale=True, random_rotate=False):
        path = pathlib.Path(root_dir)
        self.random_rotate = random_rotate
        
        if split == "train":
            filename = "train.txt"
        elif split == "val":
            filename = "val.txt"
        elif split == "test":
            filename = "test.txt"
        else:
            raise ValueError(f"Unknown split: {split}")
            
        print(f"Loading {split} data from {filename}...")
        self.files = _get_filenames(path, filename)
        
        if not self.files:
             print(f"Warning: No files found for split {split} in {root_dir}")

        self.load_graphs(self.files, center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_one_graph(self, file_path):
        sample = super().load_one_graph(file_path)
        if sample is None:
            return None
            
        try:
            # Classification datasets in this release use an integer class id
            # as the filename prefix, e.g. 003_example.bin -> label 3.
            label = int(file_path.name.split('_')[0])
            sample["label"] = torch.tensor([label]).long()
            return sample
        except Exception as e:
            print(f"Error parsing label for {file_path}: {e}")
            return None

    def _collate(self, batch):
        valid_batch = []
        
        for sample in batch:
            if sample is None or "graph" not in sample:
                continue
            g = sample["graph"]
            if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
                continue
            if 'x' not in g.ndata or 'x' not in g.edata:
                continue
            if torch.isnan(g.ndata['x']).any() or torch.isnan(g.edata['x']).any():
                continue
            
            if "label" not in sample:
                continue

            valid_batch.append(sample)

        if len(valid_batch) == 0:
            return None

        try:
            batched_graph = dgl.batch([sample["graph"] for sample in valid_batch])
            batched_filenames = [sample["filename"] for sample in valid_batch]
            batched_labels = torch.cat([sample["label"] for sample in valid_batch], dim=0)
            
            return {
                "graph": batched_graph, 
                "label": batched_labels, 
                "filename": batched_filenames
            }
            
        except Exception as e:
            print(f"Critical Error during classification collate: {e}")
            return None

class TMCADDataset(CustomBinDataset):
    @staticmethod
    def num_classes():
        return 10