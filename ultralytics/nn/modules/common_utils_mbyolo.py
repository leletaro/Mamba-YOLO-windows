import torch
import math
from functools import partial
from typing import Callable, Any

import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# ---- 原始尝试导入 CUDA 扩展（若失败将由后面的 Fallback 接管） ----
try:
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
    import selective_scan_cuda_ndstate
    import selective_scan_cuda_nrow
    # import selective_scan_cuda
except Exception:
    pass

try:
    # "sscore acts the same as mamba_ssm"
    import selective_scan_cuda_core  # noqa: F401
except Exception as e:
    print(e, flush=True)
    # "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    # import selective_scan_cuda
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


# ==================== Fallback for selective_scan (no CUDA build) ====================
_FALLBACK_ACTIVE = False
try:
    selective_scan_cuda_core  # type: ignore  # 是否已由上方导入
except NameError:
    try:
        # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _ss_fn
        from mamba_ssm.ops.selective_scan_interface import selective_scan_ref as _ss_fn

        import torch.autograd as _ag

        class _CoreFallback:
            """
            提供与 selective_scan_cuda_core 相同接口（.fwd/.bwd），
            但使用 mamba-ssm 的纯 PyTorch selective_scan_fn 实现，以便不编译也能跑。
            """

            @staticmethod
            def fwd(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, sso=1):
                # 返回签名需与 CUDA 版本一致：(out, x, *rest)
                # Fallback 不依赖 'x'，因此返回 None 占位
                out = _ss_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, sso)
                return out, None

            @staticmethod
            def bwd(u, delta, A, B, C, D, delta_bias, dout, x, delta_softplus, sso):
                # 通过重算 forward + autograd 计算梯度
                # 允许未使用的梯度（例如 D 或 delta_bias 可能为 None）
                u_ = u.detach().requires_grad_(u.requires_grad)
                delta_ = delta.detach().requires_grad_(delta.requires_grad)
                A_ = A.detach().requires_grad_(A.requires_grad)
                B_ = B.detach().requires_grad_(B.requires_grad)
                C_ = C.detach().requires_grad_(C.requires_grad)
                D_ = None if D is None else D.detach().requires_grad_(D.requires_grad)
                dbias_ = None if delta_bias is None else delta_bias.detach().requires_grad_(delta_bias.requires_grad)

                y = _ss_fn(u_, delta_, A_, B_, C_, D_, dbias_, delta_softplus, sso)

                grads = _ag.grad(
                    outputs=y,
                    inputs=(u_, delta_, A_, B_, C_, D_, dbias_),
                    grad_outputs=dout,
                    retain_graph=False,
                    allow_unused=True,
                )
                du, ddelta, dA, dB, dC, dD, ddelta_bias = grads
                return du, ddelta, dA, dB, dC, dD, ddelta_bias

        selective_scan_cuda_core = _CoreFallback()  # 替身对象，保持同名
        _FALLBACK_ACTIVE = True
        print("[Warn] Using Python fallback for selective_scan (CUDA core not found).", flush=True)

    except Exception as _e:
        raise ImportError(
            "selective_scan is unavailable: neither CUDA core nor Python fallback could be imported. "
            "Please install mamba-ssm >= 2.x (pip install mamba-ssm)."
        ) from _e
# ==================== End Fallback ====================


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# Cross Scan
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


# cross selective scan ===============================
class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x if x is not None else torch.tensor(0))
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # x 在 Fallback 下是占位张量，不参与计算；真实梯度通过重算 forward 获得
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, None, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        out_norm_shape="v0",
        nrows=-1,  # for SelectiveScanNRow
        backnrows=-1,  # for SelectiveScanNRow
        delta_softplus=True,
        to_dtype=True,
        force_fp32=False,  # False if ssoflex
        ssoflex=True,
        SelectiveScan=None,
        scan_mode_type='default'
):
    """
    out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...
    """
    # ---- 确保默认使用本文件定义的 Autograd Function ----
    if SelectiveScan is None:
        SelectiveScan = SelectiveScanCore

    B, D, H, W = x.shape
    D_state, N = A_logs.shape
    K, D_model, R = dt_projs_weight.shape
    assert D_model == D, "dt_projs_weight second dim should equal channel D"
    L = H * W

    # def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
    #     return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    import torch

    def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus,
                    nrows=None, backnrows=None, ssoflex=1):
        """
        纯 Autograd 路径：直接用 selective_scan_ref 构图，不走自定义 Function/backward。
        """
        # 兼容某些实现把 delta_bias 当 bool 开关
        if isinstance(delta_bias, (bool, int)) and not isinstance(delta_bias, torch.Tensor):
            delta_bias = None

        def _f32c(t):
            return t if not isinstance(t, torch.Tensor) else (
                t if (t.dtype == torch.float32 and t.is_contiguous())
                else t.float().contiguous()
            )

        u, delta, A, B, C, D = map(_f32c, (u, delta, A, B, C, D))

        # 直接调参考实现；不使用 z；不返回 last_state（下游只要 Tensor）
        out = _ss_fn(
            u, delta, A, B, C, D,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=bool(delta_softplus),
            return_last_state=False
        )

        # 个别版本会返回 (out, last_state)
        if isinstance(out, tuple):
            out = out[0]
        return out


    xs = CrossScan.apply(x)  # (B, 4, D, L)

    # 线性映射：得到 [R, N, N] 三段
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)

    # HiPPO matrix
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]:  # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)  # (B, H, W, C)
    else:  # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        # y = out_norm(y).view(B, H, W, -1)

        # 让输入与 LayerNorm 权重 dtype 一致，避免 "expected Float but found Half"
        if hasattr(out_norm, "weight") and isinstance(out_norm.weight, torch.Tensor):
            y = y.to(dtype=out_norm.weight.dtype)
        else:
            y = y.float()

        y = out_norm(y).view(B, H, W, -1)


    return (y.to(x.dtype) if to_dtype else y)
