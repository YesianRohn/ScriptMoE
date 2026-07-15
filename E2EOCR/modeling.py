"""
Self-contained PyTorch model definition for the Multilingual OCR recogniser.

Architecture
------------
    Input image   (B, 3, H, W)
        |
    SVTRv2-LNConv-Two33 encoder        (3-stage hybrid Conv/Attention backbone)
        |
    ScriptMoE decoder                  (Transformer decoder with MoE FFN +
                                        sample-level script-aware routing)
        |
    Logits (B, T, vocab-2)

This module reproduces *only what is needed for inference*. Any training-only
branches (load-balancing aux loss, script classifier supervision, etc.) are
intentionally dropped to keep the file small and easy to read. The state dict
key names, however, are kept identical to the original implementation, so a
checkpoint trained with the full OpenOCR-MOE codebase loads here as-is.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_


# =============================================================================
# Common building blocks
# =============================================================================
class DropPath(nn.Module):
    """Stochastic depth — no-op at eval time, kept for state-dict compatibility."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0:
            mask.div_(keep)
        return x * mask


class Identity(nn.Module):
    def forward(self, x):
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# SVTRv2 encoder (LNConv variant with two 3×3 stem convs)
# =============================================================================
class ConvBNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, bias=False, groups=1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1e-6):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = _Attention(dim, num_heads, qkv_bias, qk_scale,
                                attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class _ConvBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1e-6,
                 num_conv=2, kernel_size=3):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer = nn.Sequential(*[
            nn.Conv2d(dim, dim, kernel_size, 1,
                      kernel_size // 2, groups=num_heads)
            for _ in range(num_conv)
        ])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        C, H, W = x.shape[1:]
        x = x + self.drop_path(self.mixer(x))
        x = self.norm1(x.flatten(2).transpose(1, 2))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x


class _FlattenTranspose(nn.Module):
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class _SubSample2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3,
                              stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, C, H, W)
        return x, [H, W]


class _SubSample1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(2, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3,
                              stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, sz):
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(-1, C, sz[0], sz[1])
        x = self.conv(x)
        C, H, W = x.shape[1:]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, [H, W]


class _IdentitySize(nn.Module):
    def forward(self, x, sz):
        return x, sz


class _SVTRStage(nn.Module):
    def __init__(self, dim, out_dim, depth, mixer, kernel_sizes, sub_k,
                 num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                 attn_drop_rate, drop_path, norm_layer, act, eps,
                 num_conv, downsample):
        super().__init__()
        self.dim = dim
        self.blocks = nn.Sequential()
        for i in range(depth):
            if mixer[i] == 'Conv':
                self.blocks.append(_ConvBlock(
                    dim=dim, kernel_size=kernel_sizes[i], num_heads=num_heads,
                    mlp_ratio=mlp_ratio, drop=drop_rate, act_layer=act,
                    drop_path=drop_path[i], norm_layer=norm_layer, eps=eps,
                    num_conv=num_conv[i]))
            else:
                if mixer[i] == 'FGlobal':
                    self.blocks.append(_FlattenTranspose())
                self.blocks.append(_Block(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    act_layer=act, attn_drop=attn_drop_rate,
                    drop_path=drop_path[i], norm_layer=norm_layer, eps=eps))

        if downsample:
            if mixer[-1] == 'Conv':
                self.downsample = _SubSample2D(dim, out_dim, stride=sub_k)
            else:
                self.downsample = _SubSample1D(dim, out_dim, stride=sub_k)
        else:
            self.downsample = _IdentitySize()

    def forward(self, x, sz):
        for blk in self.blocks:
            x = blk(x)
        return self.downsample(x, sz)


class _POPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, flatten=False, bias=False):
        super().__init__()
        self.patch_embed = nn.Sequential(
            ConvBNLayer(in_channels, embed_dim // 2,
                        kernel_size=3, stride=2, padding=1, bias=bias),
            ConvBNLayer(embed_dim // 2, embed_dim,
                        kernel_size=3, stride=2, padding=1, bias=bias),
        )
        if flatten:
            self.patch_embed.append(_FlattenTranspose())

    def forward(self, x):
        sz = x.shape[2:]
        x = self.patch_embed(x)
        return x, [sz[0] // 4, sz[1] // 4]


class SVTRv2LNConvTwo33(nn.Module):
    """SVTRv2-LNConv backbone with two 3×3 stem convs.

    Default config below matches the trained checkpoint, derived from
    ``configs/rec/scriptmoe/svtrv2_scriptmoe_mlt_all.yml``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        depths=(6, 6, 6),
        dims=(128, 256, 384),
        mixer=(
            ('Conv',) * 6,
            ('Conv', 'Conv', 'FGlobal', 'Global', 'Global', 'Global'),
            ('Global',) * 6,
        ),
        sub_k=((1, 1), (2, 1), (-1, -1)),
        num_heads=(4, 8, 12),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        num_stages = len(depths)
        self.num_features = dims[-1]

        self.pope = _POPatchEmbed(
            in_channels=in_channels,
            embed_dim=dims[0],
            flatten=mixer[0][0] != 'Conv',
            bias=False,
        )

        dpr = np.linspace(0, drop_path_rate, sum(depths))
        kernel_sizes = [[3] * d for d in depths]
        num_convs = [[2] * d for d in depths]

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            stage = _SVTRStage(
                dim=dims[i],
                out_dim=dims[i + 1] if i < num_stages - 1 else 0,
                depth=depths[i],
                mixer=list(mixer[i]),
                kernel_sizes=kernel_sizes[i],
                sub_k=sub_k[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=None,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=nn.LayerNorm,
                act=nn.GELU,
                eps=eps,
                num_conv=num_convs[i],
                downsample=(i != num_stages - 1),
            )
            self.stages.append(stage)

        self.out_channels = self.num_features
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x


# =============================================================================
# ScriptMoE decoder (inference-only)
# =============================================================================
class _ExpertFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, act_layer=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _ScriptAwareRouter(nn.Module):
    """Predicts which experts to activate per sample.

    For inference we always run in 'sample' routing mode: a single set of
    experts is selected for the whole image, based on the mean-pooled encoder
    feature.
    """

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, memory):
        # memory: [B, N, D] -> routing input [B, 1, D]
        routing_input = memory.mean(dim=1, keepdim=True)
        router_logits = self.router(routing_input)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(router_probs,
                                                  self.top_k, dim=-1)
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)
        return top_k_indices, top_k_weights


class _MoEFFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts=4, top_k=2,
                 dropout=0.1, shared_expert_ratio=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        self.router = _ScriptAwareRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            _ExpertFFN(d_model, dim_feedforward, dropout)
            for _ in range(num_experts)
        ])
        shared_dim = int(dim_feedforward * shared_expert_ratio)
        self.shared_expert = _ExpertFFN(d_model, shared_dim, dropout)
        self.shared_gate = nn.Linear(d_model, 1, bias=False)

    def forward(self, x, memory):
        B, T, D = x.shape

        top_k_indices, top_k_weights = self.router(x, memory)
        # Sample-level routing -> broadcast to every token in the sequence
        top_k_indices = top_k_indices.expand(-1, T, -1)
        top_k_weights = top_k_weights.expand(-1, T, -1)

        x_flat = x.reshape(B * T, D)
        idx_flat = top_k_indices.reshape(B * T, self.top_k)
        w_flat = top_k_weights.reshape(B * T, self.top_k)

        routed = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            ek_idx = idx_flat[:, k]
            ek_w = w_flat[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (ek_idx == e)
                if mask.any():
                    routed[mask] += self.experts[e](x_flat[mask]) * ek_w[mask]
        routed = routed.reshape(B, T, D)

        shared = self.shared_expert(x)
        gate = torch.sigmoid(self.shared_gate(x))
        return gate * shared + (1 - gate) * routed


class _MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.scale = self.head_dim ** -0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        B, qN = query.shape[:2]
        if self.self_attn:
            qkv = self.qkv(query).reshape(B, qN, 3, self.num_heads,
                                          self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            kN = key.shape[1]
            q = self.q(query).reshape(B, qN, self.num_heads,
                                      self.head_dim).transpose(1, 2)
            kv = self.kv(key).reshape(B, kN, 2, self.num_heads,
                                      self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, qN, self.embed_dim)
        return self.out_proj(x)


class _MoETransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward,
                 attention_dropout_rate=0.0, residual_dropout_rate=0.1,
                 num_experts=4, top_k=2, shared_expert_ratio=0.5,
                 epsilon=1e-5):
        super().__init__()
        self.self_attn = _MultiheadAttention(d_model, nhead,
                                             attention_dropout_rate,
                                             self_attn=True)
        self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout1 = nn.Dropout(residual_dropout_rate)

        self.cross_attn = _MultiheadAttention(d_model, nhead,
                                              attention_dropout_rate,
                                              self_attn=False)
        self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout2 = nn.Dropout(residual_dropout_rate)

        self.moe_ffn = _MoEFFNLayer(
            d_model=d_model, dim_feedforward=dim_feedforward,
            num_experts=num_experts, top_k=top_k,
            dropout=residual_dropout_rate,
            shared_expert_ratio=shared_expert_ratio,
        )
        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, memory, self_mask=None):
        tgt = self.norm1(tgt + self.dropout1(
            self.self_attn(tgt, attn_mask=self_mask)))
        tgt = self.norm2(tgt + self.dropout2(
            self.cross_attn(tgt, key=memory)))
        ffn_out = self.moe_ffn(tgt, memory=memory)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))
        return tgt


class _PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float()
                             * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class _Embeddings(nn.Module):
    def __init__(self, d_model, vocab, padding_idx=0, scale_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model ** -0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        x = self.embedding(x)
        if self.scale_embedding:
            x = x * math.sqrt(self.d_model)
        return x


class ScriptMoEDecoder(nn.Module):
    """Inference-only ScriptMoE decoder (greedy autoregressive)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int = 12,
        num_decoder_layers: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        shared_expert_ratio: float = 0.5,
        max_len: int = 25,
        attention_dropout_rate: float = 0.0,
        residual_dropout_rate: float = 0.1,
        scale_embedding: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.ignore_index = out_channels - 1
        self.bos = out_channels - 2
        self.eos = 0
        self.max_len = max_len

        d_model = in_channels
        dim_feedforward = d_model * 4

        self.embedding = _Embeddings(d_model, vocab=out_channels,
                                     padding_idx=0,
                                     scale_embedding=scale_embedding)
        self.positional_encoding = _PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model)

        # Optional encoder layers in the decoder block (kept None per yml).
        self.encoder = None

        self.decoder = nn.ModuleList([
            _MoETransformerBlock(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=attention_dropout_rate,
                residual_dropout_rate=residual_dropout_rate,
                num_experts=num_experts, top_k=top_k,
                shared_expert_ratio=shared_expert_ratio,
            ) for _ in range(num_decoder_layers)
        ])

        # Auxiliary script classifier head (kept for state-dict compatibility,
        # not used at inference).
        self.script_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_experts),
        )

        self.tgt_word_prj = nn.Linear(d_model, out_channels - 2, bias=False)
        w0 = np.random.normal(0.0, d_model ** -0.5,
                              (d_model, out_channels - 2)).astype(np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())

    @staticmethod
    def _causal_mask(sz, device):
        mask = torch.triu(
            torch.full((sz, sz), float('-inf'), dtype=torch.float32),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0).to(device)

    @torch.no_grad()
    def forward(self, src):
        """Greedy autoregressive decoding.

        Args:
            src: encoder output, [B, N, D]
        Returns:
            logits over the vocabulary, [B, T, out_channels-2]
        """
        bs = src.shape[0]
        memory = src

        dec_seq = torch.full((bs, self.max_len + 1), self.ignore_index,
                             dtype=torch.int64, device=src.device)
        dec_seq[:, 0] = self.bos

        logits = []
        for step in range(self.max_len):
            embed = self.embedding(dec_seq[:, :step + 1])
            embed = self.positional_encoding(embed)
            mask = self._causal_mask(embed.shape[1], src.device)

            tgt = embed
            for layer in self.decoder:
                tgt = layer(tgt, memory, self_mask=mask)

            word_prob = F.softmax(self.tgt_word_prj(tgt[:, -1:, :]), dim=-1)
            logits.append(word_prob)
            if step < self.max_len - 1:
                dec_seq[:, step + 1] = word_prob.squeeze(1).argmax(-1)
                if (dec_seq == self.eos).any(dim=-1).all():
                    break

        return torch.cat(logits, dim=1)


# =============================================================================
# End-to-end recogniser (encoder + decoder) — wraps both above
# =============================================================================
class ScriptMoERecModel(nn.Module):
    """SVTRv2 backbone + ScriptMoE decoder.

    The state-dict layout is identical to the original
    ``BaseRecognizer(encoder=SVTRv2LNConvTwo33, decoder=ScriptMoEDecoder)``
    so checkpoints can be loaded with ``strict=False``.
    """

    def __init__(self, num_classes: int, max_len: int = 25):
        super().__init__()
        self.encoder = SVTRv2LNConvTwo33(in_channels=3, out_channels=256)
        self.decoder = ScriptMoEDecoder(
            in_channels=self.encoder.out_channels,
            out_channels=num_classes,
            nhead=12,
            num_decoder_layers=2,
            num_experts=4,
            top_k=2,
            shared_expert_ratio=0.5,
            max_len=max_len,
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder(feat)
