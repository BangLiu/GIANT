"""
We can utilize the syntactic structure of sentence
to extract style words, glue words / phrases.
"""

import torch
import torch.nn.functional as F
from .RGCN import RGCNConv
from .embedder import Embedder


class GIANTNet(torch.nn.Module):
    def __init__(self, config, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True,
                 emb_mats=None, emb_dicts=None, dropout=0.1):
        super(GIANTNet, self).__init__()
        # config
        self.config = config
        assert config.layers >= 2
        self.dropout = dropout

        # embedding
        if emb_mats is not None and len(config.emb_tags) > 0:
            self.embedder = Embedder(config, emb_mats, emb_dicts, dropout)
            self.emb_tags = config.emb_tags
            self.total_emb_size = self.embedder.get_total_emb_dim(self.emb_tags)
        else:
            self.total_emb_size = 0
        print("self.total_emb_size is: ", self.total_emb_size)

        # input conv
        self.gcn_in = RGCNConv(
            in_channels + self.total_emb_size, out_channels, num_relations, num_bases)

        # internal conv
        if config.layers > 2:
            self.gcns_internal = torch.nn.ModuleList()
            for i in range(config.layers - 2):
                self.gcns_internal.append(RGCNConv(out_channels, out_channels, num_relations, num_bases))

        # output conv
        self.gcns_out = torch.nn.ModuleList()
        for output_dim in config.task_output_dims:
            self.gcns_out.append(RGCNConv(out_channels, output_dim, num_relations, num_bases))

    def forward(self, x, emb_ids_dict, edge_index, edge_type, edge_norm=None):
        if self.total_emb_size > 0:
            # embedding input
            input_emb = self.embedder(
                emb_ids_dict, self.emb_tags).transpose(1, 2)
            # input_emb.shape: 1 * node_num * total_emb_dim
            # x.shape: 1 * node_num * total_feature_num

            # concatenated input: feature + feature embeddings
            x = torch.cat([x, input_emb], dim=-1).squeeze(0)
        else:
            x = x.squeeze(0)

        # input conv
        x = F.dropout(F.relu(self.gcn_in(x, edge_index, edge_type, edge_norm)), self.dropout, training=self.training)

        # internal conv
        if self.config.layers > 2:
            for n_l in range(self.config.layers - 2):
                x = F.dropout(F.relu(self.gcns_internal[n_l](x, edge_index, edge_type, edge_norm)), self.dropout, training=self.training)

        # output conv
        outputs = []
        for n_o in range(len(self.config.task_output_dims)):
            outputs.append(F.log_softmax(self.gcns_out[n_o](x, edge_index, edge_type, edge_norm), dim=1))

        return outputs
