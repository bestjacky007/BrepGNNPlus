import dgl
import pytorch_lightning as pl
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
import brepgnnp.encoders


class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, inp):
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


###############################################################################
# Classification
###############################################################################

class BrepGNNPClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.5,
        stochastic_depth_drop_prob=0.0,
        gnn_type='gcn',
        num_layers=12,
        no_residual=False,
        no_ffn=False,
        no_layernorm=False,
        no_stochastic_depth=False,
        no_edge=False,
    ):
        super().__init__()
        # Retain the original UV-Net geometric encoders and replace only the
        # graph-learning stage with the BrepGNN+ encoder.
        self.curv_encoder = brepgnnp.encoders.UVNetCurveEncoder(
            in_channels=6, output_dims=crv_emb_dim
        )
        self.surf_encoder = brepgnnp.encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = brepgnnp.encoders.BrepGNNPlus(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            gnn_type=gnn_type, num_layers=num_layers,
            no_residual=no_residual, no_ffn=no_ffn,
            no_layernorm=no_layernorm, no_stochastic_depth=no_stochastic_depth,
            no_edge=no_edge,
        )
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph):
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        return self.clf(graph_emb)


class Classification(pl.LightningModule):
    def __init__(self, num_classes, lr=5e-3, lr_scheduler=None, weight_decay=1e-2,
                 stochastic_depth_drop_prob=0.0, steps_per_epoch=None, gnn_type='gcn',
                 num_layers=12, no_residual=False, no_ffn=False, no_layernorm=False,
                 no_stochastic_depth=False, no_edge=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.model = BrepGNNPClassifier(
            num_classes=num_classes,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            gnn_type=gnn_type, num_layers=num_layers,
            no_residual=no_residual, no_ffn=no_ffn,
            no_layernorm=no_layernorm, no_stochastic_depth=no_stochastic_depth,
            no_edge=no_edge,
        )
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, batched_graph):
        return self.model(batched_graph)

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean", label_smoothing=0.1)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("train_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean", label_smoothing=0.1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("test_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == 'cosine':
            max_epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 100
            steps_per_epoch = self.hparams.steps_per_epoch
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch must be provided for Cosine scheduler with Warmup")
            total_steps = max_epochs * steps_per_epoch
            warmup_steps = int(0.05 * total_steps)
            if warmup_steps > 0:
                scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
                scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return optimizer


###############################################################################
# Segmentation
###############################################################################


class BrepGNNPSegmenter(nn.Module):
    def __init__(
        self,
        num_classes,
        crv_in_channels=6,
        crv_emb_dim=64,
        srf_emb_dim=64,
        graph_emb_dim=128,
        dropout=0.5,
        stochastic_depth_drop_prob=0.0,
        gnn_type='gcn',
        num_layers=12,
        no_residual=False,
        no_ffn=False,
        no_layernorm=False,
        no_stochastic_depth=False,
        no_edge=False,
    ):
        super().__init__()
        self.curv_encoder = brepgnnp.encoders.UVNetCurveEncoder(
            in_channels=crv_in_channels, output_dims=crv_emb_dim
        )
        self.surf_encoder = brepgnnp.encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = brepgnnp.encoders.BrepGNNPlus(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            gnn_type=gnn_type, num_layers=num_layers,
            no_residual=no_residual, no_ffn=no_ffn,
            no_layernorm=no_layernorm, no_stochastic_depth=no_stochastic_depth,
            no_edge=no_edge,
        )
        self.seg = _NonLinearClassifier(
            graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout
        )

    def forward(self, batched_graph):
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        node_emb, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        return self.seg(local_global_feat)


class Segmentation(pl.LightningModule):
    def __init__(self, num_classes, crv_in_channels=6, lr=5e-3, lr_scheduler=None,
                 weight_decay=1e-2, stochastic_depth_drop_prob=0.0,
                 steps_per_epoch=None, gnn_type='gcn', num_layers=12,
                 no_residual=False, no_ffn=False, no_layernorm=False,
                 no_stochastic_depth=False, no_edge=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.model = BrepGNNPSegmenter(
            num_classes, crv_in_channels=crv_in_channels,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            gnn_type=gnn_type, num_layers=num_layers,
            no_residual=no_residual, no_ffn=no_ffn,
            no_layernorm=no_layernorm, no_stochastic_depth=no_stochastic_depth,
            no_edge=no_edge,
        )
        self.train_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.val_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.test_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.train_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)
        self.test_accuracy = torchmetrics.Accuracy(num_classes=num_classes, compute_on_step=False)

    def forward(self, batched_graph):
        return self.model(batched_graph)

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean", label_smoothing=0.1)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.train_iou(preds, labels)
        self.train_accuracy(preds, labels)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_iou", self.train_iou.compute())
        self.log("train_accuracy", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean", label_smoothing=0.1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.val_iou(preds, labels)
        self.val_accuracy(preds, labels)
        return loss

    def validation_epoch_end(self, outs):
        self.log("val_iou", self.val_iou.compute())
        self.log("val_accuracy", self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.test_iou(preds, labels)
        self.test_accuracy(preds, labels)

    def test_epoch_end(self, outs):
        self.log("test_iou", self.test_iou.compute())
        self.log("test_accuracy", self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler == 'cosine':
            max_epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 100
            steps_per_epoch = self.hparams.steps_per_epoch
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch must be provided for Cosine scheduler with Warmup")
            total_steps = max_epochs * steps_per_epoch
            warmup_steps = int(0.05 * total_steps)
            if warmup_steps > 0:
                scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
                scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return optimizer
