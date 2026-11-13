import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ts_benchmark.baselines.seer.layers.linear_extractor_cluster import (
    Linear_extractor_cluster,
)
from ts_benchmark.baselines.seer.layers.graph_spectral_branch import (
    GatedResidualFusion,
    GraphSpectralBranch,
)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class AugmentedEmbedding(nn.Module):
    def __init__(self, config, d_model, patch_len, stride, padding, dropout):
        super(AugmentedEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.d_model = d_model

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.series_value_embedding = nn.Linear(config.seq_len, d_model, bias=False)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.gen1 = nn.Linear(d_model, d_model)
        self.gen2 = nn.Linear(d_model, config.d_score)
        self.gen3 = nn.Linear(d_model + config.d_score, d_model)
        self.gen4 = nn.Linear(d_model, d_model)
        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        self.cluster = Linear_extractor_cluster(config)
        self.graph_spectral_enabled = bool(
            getattr(config, "graph_spectral_enabled", False)
        )
        if self.graph_spectral_enabled:
            cache_dir = getattr(config, "graph_spectral_cache_dir", None)
            channel_order = getattr(config, "model_channel_order", None)
            if cache_dir is None:
                raise ValueError(
                    "graph_spectral_cache_dir is required when "
                    "graph_spectral_enabled=True"
                )
            if channel_order is None:
                raise ValueError(
                    "model_channel_order is required to validate graph node order "
                    "when graph_spectral_enabled=True"
                )
            self.graph_spectral_branch = GraphSpectralBranch(
                d_model=d_model,
                graph_hidden_dim=int(getattr(config, "graph_hidden_dim", 32)),
                channel_count=int(config.enc_in),
                patch_count=int((config.seq_len - patch_len) / stride + 2),
                seq_len=config.seq_len,
                patch_len=patch_len,
                stride=stride,
                graph_k=int(getattr(config, "graph_spectral_knn_k", 10)),
                cache_dir=cache_dir,
                data_path=getattr(config, "graph_spectral_data_path", None),
                channel_order=list(channel_order),
                target_channel=getattr(config, "graph_spectral_target_channel", None),
            )
            self.gated_residual_fusion = GatedResidualFusion(d_model)
        # Residual dropout
        self.dropout = nn.Dropout(p=dropout)

    def augmented_patch_embedding(self, x):
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patch_num = x.shape[2]
        z = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        shared_embedding = self.value_embedding(z)
        cluster_input = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        moe_embedding, L_importance = self.cluster(cluster_input)
        moe_embedding = moe_embedding.reshape(-1, patch_num, self.d_model)
        h_seer = moe_embedding + shared_embedding
        if self.graph_spectral_enabled:
            # The existing [B*C,N,D] layout is already channel-major and
            # therefore matches cache node_index=channel_index*N+patch_position.
            h0 = shared_embedding.reshape(x.shape[0], x.shape[1] * patch_num, self.d_model)
            h_graph = self.graph_spectral_branch(h0)
            h_seer_nodes = h_seer.reshape(
                x.shape[0], x.shape[1] * patch_num, self.d_model
            )
            h_fused = self.gated_residual_fusion(h_seer_nodes, h_graph)
            h_seer = h_fused.reshape(-1, patch_num, self.d_model)
        # Position embedding remains added exactly once, after branch fusion.
        patch_embedding = h_seer + self.position_embedding(z)
        return patch_embedding, L_importance

    def augmented_series_embedding(self, x_enc, x_mark):
        series_emb = self.series_value_embedding(
            torch.cat([x_enc, x_mark.permute(0, 2, 1)], 1)
        )
        series_emb = self.dropout(series_emb)
        combined_mean = F.gelu(self.gen1(series_emb))
        combined_mean = self.gen2(combined_mean)

        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, series_emb.size(1))
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(series_emb.size(0), -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, series_emb.size(1), 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(
                combined_mean * weight, dim=1, keepdim=True
            ).repeat(1, series_emb.size(1), 1)

        combined_mean_cat = torch.cat([series_emb, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        series_global_emb = self.gen4(combined_mean_cat)
        series_global_emb = series_global_emb[:, : x_enc.size(1), :]

        series_global_emb = series_global_emb.reshape(-1, 1, series_global_emb.size(-1))
        return series_global_emb

    def forward(self, x_enc, x_mark):
        patch_embedding, l_importance = self.augmented_patch_embedding(x_enc)
        series_embedding = self.augmented_series_embedding(x_enc, x_mark)

        reshaped_output = torch.cat([series_embedding, patch_embedding], dim=1)

        return reshaped_output, l_importance
