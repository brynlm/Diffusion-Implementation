import torch
import torch.nn as nn


# Taken from explaining-AI's repo:
def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    t_emb.to(time_steps.device)
    return t_emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_norm_groups, num_attn_heads, 
                 self_attn=False, use_context=False, down_sample=False, up_sample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_norm_groups = num_norm_groups
        self.num_attn_heads = num_attn_heads
        self.self_attn = self_attn
        self.use_context = use_context
        self.down_sample = down_sample
        self.up_sample = up_sample

        self.res1 = nn.Sequential(
            nn.GroupNorm(num_norm_groups if num_norm_groups % in_channels == 0 else 1, in_channels),
            nn.SiLU(),
            nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
        )

        self.res_proj = nn.LazyConv2d(out_channels, kernel_size=1)

        self.res2 = nn.Sequential(
            nn.GroupNorm(num_norm_groups, out_channels),
            nn.SiLU(),
            nn.LazyConv2d(out_channels, kernel_size=3, padding=1)
        )

        self.t_emb = nn.Sequential(
            nn.SiLU(),
            nn.LazyLinear(out_channels)
        )
        if self_attn:
            self.attn_norm = nn.GroupNorm(num_norm_groups, out_channels)
            self.attn = nn.MultiheadAttention(out_channels, num_attn_heads, batch_first=True)
        
        if use_context:
            self.cross_attn_norm = nn.GroupNorm(num_norm_groups, out_channels)
            self.cross_attn = nn.MultiheadAttention(out_channels, num_attn_heads, batch_first=True)
            self.context_proj = nn.LazyConv2d(out_channels, kernel_size=1)

        if down_sample:
            self.downsamp = nn.LazyConv2d(in_channels, kernel_size=4, stride=2, padding=1)

        if up_sample:
            self.upsamp = nn.LazyConvTranspose2d(out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t, down_output=None, context=None):
        # Upsample
        if self.up_sample:
            x = self.upsamp(x)
            if down_output is not None:
                x = torch.cat([x, down_output], dim=1) # Concatenate along channel dimension

        # Downsample
        if self.down_sample:
            x = self.downsamp(x)

        out = self.res1(x) + self.t_emb(t)[:, :, None, None]    # First conv layer + time encoding
        out = self.res2(out) + self.res_proj(x)     # Second conv layer + skip connection
        
        # Self-attention layer + skip connection
        if self.attn:
            n, c, h, w = out.shape
            attn_in = out.reshape(n, c, h*w)
            attn_in = self.attn_norm(attn_in)
            attn_in = attn_in.transpose(1, 2) # MHA layer expects shape (N, S, C)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False) 
            attn_out = attn_out.transpose(1, 2).reshape(n, c, h, w)
            out = out + attn_out

        # Cross-attention layer + skip connection
        if self.use_context:
            n, c, h, w = out.shape
            attn_in = out.reshape(n, c, h*w)
            attn_in = self.cross_attn_norm(attn_in)
            attn_in = attn_in.transpose(1, 2)

            context_proj = self.context_proj(context)
            n_k, c_k, h_k, w_k = context_proj.shape
            cross_attn_in = context_proj.reshape(n_k, c_k, h_k*w_k).transpose(1, 2)
            attn_out, _ = self.cross_attn(attn_in, cross_attn_in, cross_attn_in, need_weights=False)
            attn_out = attn_out.transpose(1, 2).reshape(n, c, h, w)
            out = out + attn_out

        return out