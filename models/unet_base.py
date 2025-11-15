import torch
import torch.nn as nn
from .res_block import ResBlock, get_time_embedding


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_norm_groups, num_down_blocks, num_mid_blocks, use_context=False):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.in_channels = in_channels    
        self.out_channels = out_channels
        self.num_down_blocks = num_down_blocks
        if isinstance(out_channels, int):
            self.out_channels = [out_channels] * num_down_blocks
        self.num_mid_blocks = num_mid_blocks

        self.down_blocks = nn.ModuleList([
            ResBlock(self.out_channels[0] if i==0 else self.out_channels[i-1],#in_channels if i==0 else self.out_channels[i-1], 
                     num_channels, 
                     num_norm_groups, 
                     num_heads, 
                     self_attn=True,
                     down_sample=False if i==0 else True,
                     use_context=use_context
            ) 
            for i, num_channels in enumerate(self.out_channels)
        ])
        self.mid_blocks = nn.ModuleList([
            ResBlock(self.out_channels[-1], 
                     self.out_channels[-1], 
                     num_norm_groups, 
                     num_heads, 
                     self_attn=True,
                     down_sample=True if i==0 else False,
                     use_context=use_context
            ) 
            for i in range(num_mid_blocks)
        ])
        self.up_blocks = nn.ModuleList([
            ResBlock(num_channels*2, 
                     num_channels,
                     num_norm_groups, 
                     num_heads, 
                     self_attn=True,
                     up_sample=True
            ) 
            for num_channels in reversed(self.out_channels)
        ])

        self.input_conv = nn.Conv2d(
            in_channels, 
            out_channels=self.out_channels[0], 
            kernel_size=3, 
            padding=1
        )

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_norm_groups, self.out_channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.out_channels[0], in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, context=None):
        t_emb = get_time_embedding(t, self.t_emb_dim)
        down_outputs = []

        out = self.input_conv(x)
        for down_block in self.down_blocks:
            out = down_block(out, t_emb, context=context)
            down_outputs.append(out)
        
        for mid_block in self.mid_blocks:
            out = mid_block(out, t_emb, context=context)
        
        for up_block in self.up_blocks:
            down_output = down_outputs.pop()
            out = up_block(out, t_emb, down_output=down_output)

        out = self.output_conv(out)
        return out