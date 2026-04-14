import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.glob import MaxPooling, SumPooling
import dgl.function as fn


# =============================================================================
# Utility: DropPath (Stochastic Depth)
# =============================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# =============================================================================
# UV-Net Geometric Encoders
# =============================================================================

# These two geometric encoders are retained from the original UV-Net design,
# because BrepGNN+ fixes the geometric encoding stage and modernizes only the
# graph-learning stage. The implementations here are adapted from the
# MIT-licensed UV-Net repository and are kept separate from the BrepGNN+
# graph encoder for clarity and attribution.

def _conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(),
    )


def _conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )

def _fc(in_features, out_features, bias=False):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )




class UVNetCurveEncoder(nn.Module):
    def __init__(self, in_channels=6, output_dims=64):
        """
        This is the 1D convolutional network that extracts features from the B-rep edge
        geometry described as 1D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         curve tangents. Defaults to 6.
            output_dims (int, optional): Output curve embedding dimension. Defaults to 64.
        """
        super(UVNetCurveEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.conv2 = _conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.conv3 = _conv1d(128, 256, kernel_size=3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, output_dims, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x




class UVNetSurfaceEncoder(nn.Module):
    def __init__(
        self,
        in_channels=7,
        output_dims=64,
    ):
        """
        This is the 2D convolutional network that extracts features from the B-rep face
        geometry described as 2D UV-grids (see Section 3.2, Curve & surface convolution
        in paper)

        Args:
            in_channels (int, optional): Number of channels in the edge UV-grids. By default
                                         we expect 3 channels for point coordinates and 3 for
                                         surface normals and 1 for the trimming mask. Defaults
                                         to 7.
            output_dims (int, optional): Output surface embedding dimension. Defaults to 64.
        """
        super(UVNetSurfaceEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv1 = _conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.conv2 = _conv2d(64, 128, 3, padding=1, bias=False)
        self.conv3 = _conv2d(128, 256, 3, padding=1, bias=False)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, output_dims, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        assert x.size(1) == self.in_channels
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


# =============================================================================
# GNN Layers with full ablation support
# =============================================================================

class BaseGNNLayer(nn.Module):
    """
    Base GNN layer with toggleable components for ablation:
      - residual connection
      - FFN (feed-forward network)
      - LayerNorm (can be replaced with Identity)
      - Stochastic Depth (DropPath)
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, residual=True, ffn=True,
                 drop_path=0.0, use_layernorm=True):
        super().__init__()
        self.residual = residual
        self.ffn = ffn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Norm layers (can be disabled for ablation)
        if use_layernorm:
            self.norm_node = nn.LayerNorm(out_dim)
            self.norm_edge = nn.LayerNorm(out_dim)
        else:
            self.norm_node = nn.Identity()
            self.norm_edge = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        if self.ffn:
            if use_layernorm:
                self.norm1_local = nn.LayerNorm(out_dim)
                self.norm2 = nn.LayerNorm(out_dim)
                self.norm_e2 = nn.LayerNorm(out_dim)
            else:
                self.norm1_local = nn.Identity()
                self.norm2 = nn.Identity()
                self.norm_e2 = nn.Identity()
            self.ff_linear1 = nn.Linear(out_dim, out_dim * 2)
            self.ff_linear2 = nn.Linear(out_dim * 2, out_dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, g, h, e):
        h_in, e_in = h, e
        h_new, e_new = self.message_passing(g, h, e)

        h_new = self.norm_node(h_new)
        e_new = self.norm_edge(e_new)
        h_new = F.relu(h_new)
        e_new = F.relu(e_new)
        h_new = self.dropout(h_new)
        e_new = self.dropout(e_new)

        if self.residual:
            h_new = h_in + self.drop_path(h_new)
            e_new = e_in + self.drop_path(e_new)

        if self.ffn:
            h_new = h_new + self.drop_path(self._ff_block(self.norm1_local(h_new)))
            h_new = self.norm2(h_new)
            e_new = self.norm_e2(e_new)

        return h_new, e_new

    def _ff_block(self, x):
        x = self.ff_dropout1(F.relu(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def message_passing(self, g, h, e):
        raise NotImplementedError


class GCNLayer(BaseGNNLayer):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.linear_h = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_e = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_e_update = nn.Linear(in_dim, out_dim, bias=True)

    def message_passing(self, g, h, e):
        with g.local_scope():
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(h.device).unsqueeze(1)
            h_src = self.linear_h(h) * norm
            e_msg = self.linear_e(e)
            g.ndata['h'] = h_src
            g.edata['e'] = e_msg
            
            g.update_all(lambda edges: {'m': edges.src['h'] + edges.data['e']}, fn.sum('m', 'h_agg'))
            h_new = g.ndata['h_agg'] * norm
            e_new = self.linear_e_update(e)
            return h_new, e_new


class GraphSAGELayer(BaseGNNLayer):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.linear_h = nn.Linear(in_dim * 2, out_dim, bias=True)
        self.linear_e_update = nn.Linear(in_dim, out_dim, bias=True)

    def message_passing(self, g, h, e):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['e'] = e
            
            g.update_all(lambda edges: {'m': edges.src['h'] + edges.data['e']}, fn.mean('m', 'h_agg'))
            h_agg = g.ndata['h_agg']
            h_new = self.linear_h(torch.cat([h, h_agg], dim=1))
            e_new = self.linear_e_update(e)
            return h_new, e_new


class GINLayer(BaseGNNLayer):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__(in_dim, out_dim, **kwargs)
        self.mlp = nn.Linear(in_dim, out_dim, bias=True)
        self.eps = nn.Parameter(torch.zeros(1))
        self.linear_e_update = nn.Linear(in_dim, out_dim, bias=True)

    def message_passing(self, g, h, e):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['e'] = e
            
            g.update_all(lambda edges: {'m': edges.src['h'] + edges.data['e']}, fn.sum('m', 'h_agg'))
            h_agg = g.ndata['h_agg']
            h_new = self.mlp((1 + self.eps) * h + h_agg)
            e_new = self.linear_e_update(e)
            return h_new, e_new


class GatedGCNLayer(nn.Module):
    """
    Full GatedGCN layer with ablation toggles for residual, FFN, LayerNorm,
    Stochastic Depth, and edge features.
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, residual=True, ffn=True,
                 drop_path=0.0, use_layernorm=True):
        super().__init__()
        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)

        if use_layernorm:
            self.bn_node_x = nn.LayerNorm(out_dim)
            self.bn_edge_e = nn.LayerNorm(out_dim)
        else:
            self.bn_node_x = nn.Identity()
            self.bn_edge_e = nn.Identity()

        self.dropout = dropout
        self.residual = residual
        self.ffn = ffn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            if use_layernorm:
                self.norm1_local = nn.LayerNorm(out_dim)
                self.norm2 = nn.LayerNorm(out_dim)
                self.norm_e2 = nn.LayerNorm(out_dim)
            else:
                self.norm1_local = nn.Identity()
                self.norm2 = nn.Identity()
                self.norm_e2 = nn.Identity()
            self.ff_linear1 = nn.Linear(out_dim, out_dim * 2)
            self.ff_linear2 = nn.Linear(out_dim * 2, out_dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def edge_gate_func(self, edges):
        Dx = edges.dst['Dx']
        Ex = edges.src['Ex']
        Ce = edges.data['Ce']
        e_ij = Dx + Ex + Ce
        sigma_ij = torch.sigmoid(e_ij)
        return {'e': e_ij, 'sigma': sigma_ij}

    def forward(self, g, h, efeat):
        with g.local_scope():
            if self.residual:
                h_in = h
                e_in = efeat

            Ax = self.A(h)
            Bx = self.B(h)
            Dx = self.D(h)
            Ex = self.E(h)
            Ce = self.C(efeat)

            g.ndata['Bx'] = Bx
            g.ndata['Dx'] = Dx
            g.ndata['Ex'] = Ex
            g.edata['Ce'] = Ce

            g.apply_edges(self.edge_gate_func)
            g.update_all(lambda edges: {'m_sigma_Bx': edges.src['Bx'] * edges.data['sigma']}, fn.sum('m_sigma_Bx', 'sum_sigma_Bx'))
            g.update_all(lambda edges: {'m_sigma': edges.data['sigma']}, fn.sum('m_sigma', 'sum_sigma'))

            numerator = g.ndata['sum_sigma_Bx']
            denominator = g.ndata['sum_sigma']
            out = numerator / (denominator + 1e-6)

            h = Ax + out
            e_out = g.edata['e']

            h = self.bn_node_x(h)
            e_out = self.bn_edge_e(e_out)
            h = F.relu(h)
            e_out = F.relu(e_out)
            h = F.dropout(h, self.dropout, training=self.training)
            e_out = F.dropout(e_out, self.dropout, training=self.training)

            if self.residual:
                h = h_in + self.drop_path(h)
                e_out = e_in + self.drop_path(e_out)

            if self.ffn:
                h = h + self.drop_path(self._ff_block(self.norm1_local(h)))
                h = self.norm2(h)
                e_out = self.norm_e2(e_out)

            return h, e_out

    def _ff_block(self, x):
        x = self.ff_dropout1(F.relu(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))


# =============================================================================
# Graph Encoder
# =============================================================================

class BrepGNNPlus(nn.Module):
    """
    BrepGNN+ graph encoder supporting multiple GNN types and full ablation
    controls.

    Ablation flags:
      - gnn_type: gcn, sage, gin, gatedgcn
      - no_residual: disable residual connections
      - no_ffn: disable feed-forward networks
      - no_layernorm: replace LayerNorm with Identity
      - no_stochastic_depth: set all drop_path to 0
      - no_edge: zero out edge features (keep parameters)
    """
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        num_layers=12,
        dropout=0.1,
        stochastic_depth_drop_prob=0.0,
        gnn_type='gcn',
        no_residual=False,
        no_ffn=False,
        no_layernorm=False,
        no_stochastic_depth=False,
        no_edge=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.no_edge = no_edge

        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(input_edge_dim, hidden_dim)

        # Stochastic depth schedule
        if no_stochastic_depth:
            dpr = [0.0] * num_layers
        else:
            dpr = [x.item() for x in torch.linspace(0, stochastic_depth_drop_prob, num_layers)]

        use_residual = not no_residual
        use_ffn = not no_ffn
        use_layernorm = not no_layernorm

        layer_map = {
            'gcn': GCNLayer,
            'sage': GraphSAGELayer,
            'gin': GINLayer,
            'gatedgcn': GatedGCNLayer,
        }
        if gnn_type not in layer_map:
            raise ValueError(f"Unknown GNN type: {gnn_type}. Choose from {list(layer_map.keys())}")
        Layer = layer_map[gnn_type]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                Layer(hidden_dim, hidden_dim, dropout=dropout, residual=use_residual,
                      ffn=use_ffn, drop_path=dpr[i], use_layernorm=use_layernorm)
            )

        self.head = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.pool1 = SumPooling()
        self.pool2 = MaxPooling()

    def forward(self, g, h, efeat):
        x = self.node_encoder(h)
        e = self.edge_encoder(efeat)

        # Ablation: zero out edge features while keeping parameters
        if self.no_edge:
            e = torch.zeros_like(e)

        for layer in self.layers:
            x, e = layer(g, x, e)

        graph_emb1 = self.pool1(g, x)
        graph_emb2 = self.pool2(g, x)
        graph_emb = torch.cat([graph_emb1, graph_emb2], dim=-1)
        out = self.head(graph_emb)
        return x, out
