import torch
import torch.nn as nn

class MSDeformAttn(nn.Module):
    """
    Lightweight Deformable Attention Module (DAM)
    compatible with PARSeq encoder output.
    This is NOT heavy Deformable DETR code.
    It is intentionally simple and stable.
    """

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multi-head attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward refinement
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x