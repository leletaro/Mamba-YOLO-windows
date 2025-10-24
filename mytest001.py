import mamba_ssm.ops.selective_scan_interface as ssi
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
ssi.selective_scan_fn = selective_scan_ref

import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print('success')