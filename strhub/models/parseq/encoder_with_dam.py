import torch.nn as nn
from .modules import Encoder               # PARSeq's original encoder (unchanged)
from .ms_deform_attn import MSDeformAttn   # Your new C-module DAM

class EncoderWithDAM(nn.Module):
    """
    Wraps PARSeq's original Encoder and applies MSDeformAttn
    to enhance visual features (blur, occlusion, noise, scratches, etc.)
    """

    def __init__(self, img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio):
        super().__init__()

        # Original ViT Encoder — NO CHANGES
        self.encoder = Encoder(
            img_size,
            patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Your new DAM module
        self.dam = MSDeformAttn(embed_dim, num_heads=num_heads)

    def forward(self, x):
        # ViT encoder output
        memory = self.encoder(x)

        # Apply DAM enhancement on feature tokens
        memory = self.dam(memory)

        return memory