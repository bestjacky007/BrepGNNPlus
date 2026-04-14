import argparse
import pathlib
import time

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from datasets.solidletters import SolidLetters
from datasets.classification_datasets import TMCADDataset
from brepgnnp.models import Classification

# ======================== Argument Parser ========================
parser = argparse.ArgumentParser("BrepGNN+ Classification")
parser.add_argument("traintest", choices=("train", "test"), help="Train or test")
parser.add_argument("--dataset", choices=("solidletters", "tmcad"), required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--experiment_name", type=str, default="classification")
parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=("cosine",))
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-2)
# parser.add_argument("--seed", type=int, default=42)

# GNN Architecture
parser.add_argument("--gnn_type", type=str, default="gcn",
                    choices=("gcn", "sage", "gin", "gatedgcn"))
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--stochastic_depth_drop_prob", type=float, default=0.1)

# Ablation flags
parser.add_argument("--no_residual", action="store_true", help="Ablation: remove residual connections")
parser.add_argument("--no_ffn", action="store_true", help="Ablation: remove FFN blocks")
parser.add_argument("--no_layernorm", action="store_true", help="Ablation: replace LayerNorm with Identity")
parser.add_argument("--no_stochastic_depth", action="store_true", help="Ablation: disable Stochastic Depth")
parser.add_argument("--no_edge", action="store_true", help="Ablation: zero out edge features")

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()


# ======================== Callbacks ========================
class PeriodicCheckpoint(Callback):
    def __init__(self, start_epoch, interval, save_path):
        super().__init__()
        self.start_epoch = start_epoch
        self.interval = interval
        self.save_path = save_path

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch >= self.start_epoch and epoch % self.interval == 0:
            filename = f"epoch={epoch:02d}-periodic.ckpt"
            file_path = str(pathlib.Path(self.save_path).joinpath(filename))
            trainer.save_checkpoint(file_path)


# ======================== Setup ========================
results_path = pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
results_path.mkdir(parents=True, exist_ok=True)

month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
ckpt_dir = str(results_path.joinpath(month_day, hour_min_second))

checkpoint_callback_loss = ModelCheckpoint(
    monitor="val_loss", dirpath=ckpt_dir,
    filename="best-loss-{epoch:02d}-{val_loss:.4f}", save_top_k=1, mode='min', save_last=True,
)
checkpoint_callback_accuracy = ModelCheckpoint(
    monitor='val_acc', dirpath=ckpt_dir,
    filename='best-accuracy-{epoch:02d}-{val_acc:.4f}', save_top_k=1, mode='max'
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback_loss, checkpoint_callback_accuracy],
    logger=TensorBoardLogger(str(results_path), name=month_day, version=hour_min_second),
    resume_from_checkpoint=args.checkpoint,
)

# ======================== Dataset ========================
DATASETS = {
    "solidletters": SolidLetters,
    "tmcad": TMCADDataset,
}
Dataset = DATASETS[args.dataset]

# ======================== Train / Test ========================
if args.traintest == "train":
    seed_everything(workers=True)

    # Build ablation description
    ablation_flags = []
    if args.no_residual: ablation_flags.append("no_residual")
    if args.no_ffn: ablation_flags.append("no_ffn")
    if args.no_layernorm: ablation_flags.append("no_layernorm")
    if args.no_stochastic_depth: ablation_flags.append("no_stochastic_depth")
    if args.no_edge: ablation_flags.append("no_edge")
    ablation_str = ", ".join(ablation_flags) if ablation_flags else "None"

    print(f"""
================================================================================
BrepGNN+ Classification
================================================================================
Dataset:       {args.dataset}
GNN Type:      {args.gnn_type}
Num Layers:    {args.num_layers}
LR:            {args.lr}
Seed:          {args.seed if hasattr(args, 'seed') else 'None'}
Ablations:     {ablation_str}
Logs:          results/{args.experiment_name}/{month_day}/{hour_min_second}
================================================================================
    """)

    train_data = Dataset(root_dir=args.dataset_path, split="train")
    val_data = Dataset(root_dir=args.dataset_path, split="val")
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Classification(
        num_classes=Dataset.num_classes(),
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        weight_decay=args.weight_decay,
        stochastic_depth_drop_prob=args.stochastic_depth_drop_prob,
        steps_per_epoch=len(train_loader),
        gnn_type=args.gnn_type,
        num_layers=args.num_layers,
        no_residual=args.no_residual,
        no_ffn=args.no_ffn,
        no_layernorm=args.no_layernorm,
        no_stochastic_depth=args.no_stochastic_depth,
        no_edge=args.no_edge,
    )
    trainer.fit(model, train_loader, val_loader)

else:
    assert args.checkpoint is not None, "Expected --checkpoint for testing"
    test_data = Dataset(root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = Classification.load_from_checkpoint(args.checkpoint)
    results = trainer.test(model=model, test_dataloaders=[test_loader], verbose=False)
    print(f"Classification accuracy (%) on test set: {results[0]['test_acc'] * 100.0:.4f}")
