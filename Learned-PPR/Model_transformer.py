"""
@author: Yoni Choukroun, choukroun.yoni@gmail.com
Error Correction Code Transformer
https://arxiv.org/abs/2203.14966
"""

from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from Codes import sign_to_bin
import torch
import torch.nn as nn
from performer_pytorch import Performer
from positional_encodings.torch_encodings import PositionalEncoding2D


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


############################################################


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        ####
        code = args.code
        self.n = code.n
        self.k = code.k
        self.l = code.l

        self.pc_matrix_size_0 = code.l * (code.n - code.k)
        self.src_embed = torch.nn.Parameter(
            torch.empty((code.n * code.l + self.pc_matrix_size_0, args.d_model))
        )

        self.src_embed_T = torch.nn.Parameter(
            torch.empty((code.n * code.l + self.pc_matrix_size_0, args.d_model))
        )
        self.d_model = args.d_model

        self.decoder = nn.ModuleList(
            [
                Performer(
                    dim=args.d_model,
                    depth=1,
                    heads=args.h,
                    dim_head=args.d_model // args.h,
                    ff_mult=2,
                    reversible=True,  # 启用参数共享
                    feature_redraw_interval=5000,  # 特征重绘间隔
                )
                for _ in range(args.N_dec)
            ]
        )
        self.decoder_T = nn.ModuleList(
            [
                Performer(
                    dim=args.d_model,
                    depth=1,
                    heads=args.h,
                    dim_head=args.d_model // args.h,
                    ff_mult=2,
                    reversible=True,  # 启用参数共享
                    feature_redraw_interval=5000,  # 特征重绘间隔
                )
                for _ in range(args.N_dec)
            ]
        )

        self.oned_final_embed = torch.nn.Sequential(*[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(
            code.n * code.l + self.pc_matrix_size_0, code.n * code.l
        )

        self.get_mask(code, no_mask=True)
        logging.info(f"Mask:\n {self.src_mask}")
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome, magnitude_T, syndrome_T, mask=None):
        bs, _, _ = magnitude.size()
        emb = (
            torch.cat([magnitude, syndrome], -1)
            .view(bs, self.l * (2 * self.n - self.k))
            .unsqueeze(-1)
        )
        emb_T = (
            torch.cat([magnitude_T, syndrome_T], 1)
            .view(bs, (2 * self.n - self.k) * self.l)
            .unsqueeze(-1)
        )
        emb = self.src_embed.unsqueeze(0) * emb
        emb_T = self.src_embed_T.unsqueeze(0) * emb_T
        # emb = self.decoder(emb)
        for layer in self.decoder:
            emb = layer(emb)

        for layer in self.decoder_T:
            emb_T = layer(emb_T)
        emb_T = (
            emb_T.contiguous()
            .view(bs, (2 * self.n - self.k), self.l, self.d_model)
            .transpose(1, 2)
            .contiguous()
            .view(bs, (2 * self.n - self.k) * self.l, self.d_model)
        )

        emb = emb + emb_T
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z2)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            pc_matrix = code.pc_matrix[: code.n - code.k, : code.n]
            mask_size = code.n + pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(pc_matrix.size(0)):
                idx = torch.where(pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask = build_mask(code)
        I_l = torch.eye(self.d_model)
        src_mask = torch.kron(I_l, src_mask).bool()
        mask_size = (code.n + code.pc_matrix.size(0)) * code.l
        a = mask_size**2
        logging.info(
            f"Self-Attention Sparsity Ratio={100 * torch.sum((src_mask).int()) / a:0.2f}%, Self-Attention Complexity Ratio={100 * torch.sum((~src_mask).int())//2 / a:0.2f}%"
        )
        self.register_buffer("src_mask", src_mask)


class ECC_Transformer_baseline(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer_baseline, self).__init__()
        ####
        code = args.code
        self.n = code.n
        self.k = code.k
        self.l = code.l

        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model * 4, dropout)

        self.pc_matrix_size_0 = code.l * (code.n - code.k)
        self.src_embed = torch.nn.Parameter(
            torch.empty((code.n * code.l + self.pc_matrix_size_0, args.d_model))
        )

        self.src_embed_T = torch.nn.Parameter(
            torch.empty((code.n * code.l + self.pc_matrix_size_0, args.d_model))
        )
        self.d_model = args.d_model

        # self.decoder = nn.ModuleList(
        #     [MultiHeadedAttention(args.h, args.d_model) for _ in range(args.N_dec)]
        # )
        self.decoder = Encoder(
            EncoderLayer(args.d_model, c(attn), c(ff), dropout), args.N_dec
        )

        # self.decoder_T = nn.ModuleList(
        #     [MultiHeadedAttention(args.h, args.d_model) for _ in range(args.N_dec)]
        # )

        self.decoder_T = Encoder(
            EncoderLayer(args.d_model, c(attn), c(ff), dropout), args.N_dec
        )

        self.oned_final_embed = torch.nn.Sequential(*[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(
            code.n * code.l + self.pc_matrix_size_0, code.n * code.l
        )

        self.get_mask(code, no_mask=True)
        logging.info(f"Mask:\n {self.src_mask}")
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome, magnitude_T, syndrome_T, mask=None):
        bs, _, _ = magnitude.size()
        emb = (
            torch.cat([magnitude, syndrome], -1)
            .view(bs, self.l * (2 * self.n - self.k))
            .unsqueeze(-1)
        )
        emb_T = (
            torch.cat([magnitude_T, syndrome_T], 1)
            .view(bs, (2 * self.n - self.k) * self.l)
            .unsqueeze(-1)
        )
        emb = self.src_embed.unsqueeze(0) * emb
        emb_T = self.src_embed_T.unsqueeze(0) * emb_T

        emb = self.decoder(emb)
        emb_T = self.decoder_T(emb_T)
        # for layer in self.decoder:
        #     emb = layer(emb)

        # for layer in self.decoder_T:
        #     emb_T = layer(emb_T)
        emb_T = (
            emb_T.contiguous()
            .view(bs, (2 * self.n - self.k), self.l, self.d_model)
            .transpose(1, 2)
            .contiguous()
            .view(bs, (2 * self.n - self.k) * self.l, self.d_model)
        )

        emb = emb + emb_T
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(z_pred, sign_to_bin(torch.sign(z2)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            pc_matrix = code.pc_matrix[: code.n - code.k, : code.n]
            mask_size = code.n + pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(pc_matrix.size(0)):
                idx = torch.where(pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~(mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask = build_mask(code)
        I_l = torch.eye(self.d_model)
        src_mask = torch.kron(I_l, src_mask).bool()
        mask_size = (code.n + code.pc_matrix.size(0)) * code.l
        a = mask_size**2
        logging.info(
            f"Self-Attention Sparsity Ratio={100 * torch.sum((src_mask).int()) / a:0.2f}%, Self-Attention Complexity Ratio={100 * torch.sum((~src_mask).int())//2 / a:0.2f}%"
        )
        self.register_buffer("src_mask", src_mask)


############################################################
############################################################

if __name__ == "__main__":
    from thop import profile, clever_format
    import time
    import numpy as np
    import argparse

    def measure_latency(
        model,
        input_shape,
        device="cuda",
        dtype=torch.float32,
        warmup=50,
        runs=200,
        batch_size=1,
        verbose=True,
    ):
        model.to(device).eval()
        k, n, l = input_shape
        # create dummy input
        magnitute = torch.randn((batch_size, l, n), dtype=dtype, device=device)
        syndrome = torch.randn((batch_size, l, (n - k)), dtype=dtype, device=device)
        magnitute_T = torch.randn((batch_size, n, l), dtype=dtype, device=device)
        syndrome_T = torch.randn((batch_size, (n - k), l), dtype=dtype, device=device)
        # warm-up
        with torch.no_grad():
            for _ in range(warmup):
                flops, params = profile(
                    model, inputs=(magnitute, syndrome, magnitute_T, syndrome_T)
                )
                flops, params = clever_format([flops, params], "%.3f")
                print("model params: ", params)
                print("flops is: ", flops)
                out = model(magnitute, syndrome, magnitute_T, syndrome_T)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
        # timed runs
        times = []
        with torch.no_grad():
            for i in range(runs):
                start = time.perf_counter()
                out = model(magnitute, syndrome, magnitute_T, syndrome_T)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) / batch_size)  # per-sample latency
        times = np.array(times)
        if verbose:
            print(f"device={device}, batch_size={batch_size}, runs={runs}")
            print(f"mean(ms)  : {times.mean()*1000:.3f}")
            print(f"median(ms): {np.median(times)*1000:.3f}")
            print(f"p90(ms)   : {np.percentile(times,90)*1000:.3f}")
            print(f"p95(ms)   : {np.percentile(times,95)*1000:.3f}")
            print(f"p99(ms)   : {np.percentile(times,99)*1000:.3f}")
            print(f"std(ms)   : {times.std()*1000:.3f}")
            total_time = times.sum() * batch_size
            throughput = (runs * batch_size) / (times.sum())
            print(f"throughput (samples/sec): {throughput:.2f}")
        return times

    k = 16
    n = 24
    m = 50

    parser = argparse.ArgumentParser(description="PyTorch ECCT")
    # Code args
    parser.add_argument("--code_l", type=int, default=50)
    parser.add_argument("--code_k", type=int, default=16)
    parser.add_argument("--code_n", type=int, default=24)

    # model args
    parser.add_argument("--N_dec", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--h", type=int, default=8)

    args = parser.parse_args()

    class Code:
        pass

    code = Code()
    code.l = args.code_l
    code.k = args.code_k
    code.n = args.code_n
    args.code = code

    model = ECC_Transformer_baseline(args)
    measure_latency(
        model,
        input_shape=(k, n, m),
        device="cuda",
        batch_size=32,
        warmup=10,
        runs=300,
    )
