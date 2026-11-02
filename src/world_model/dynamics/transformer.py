import torch
import torch.nn as nn
import torch.nn.functional as F


class FunctionTransformer(nn.Module):
    """
    Structural function learning module supporting variable-length Pa(Xi) sets:
    - Each parent variable -> one token (containing variable ID and value)
    - Supports variable-length input (mask invalid positions)
    - Captures interaction relationships between parent variables through Transformer
    - Outputs target variable prediction
    """

    def __init__(self, num_vars: int, model_dim: int = 128, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super(FunctionTransformer, self).__init__()

        self.model_dim = model_dim
        self.num_vars = num_vars

        # Embedding layer: variable type + value
        self.var_embed = nn.Embedding(num_vars, model_dim)
        self.val_proj = nn.Linear(1, model_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer (predict target value after aggregation)
        self.output_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 1)
        )

    def forward(self, values, var_ids=None, mask=None):
        """
        values: [B, L, P] or [B, N] or [B, N, 1]
        """
        B = values.size(0)

        # [FIX] Support [B, L, P] -> [B, N, 1]
        if values.dim() == 3:
            B, L, P = values.shape
            values = values.reshape(B, L * P, 1)
        elif values.dim() == 2:
            values = values.unsqueeze(-1)

        N = values.size(1)

        # [Key Fix] var_ids out of bounds issue: always limit index range to [0, num_vars-1]
        device = values.device
        if var_ids is None:
            var_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        var_ids = var_ids % self.num_vars   # Prevent index out of range

        # Construct token embedding
        token_embed = self.var_embed(var_ids) + self.val_proj(values)  # [B, N, D]

        # Transformer encoding
        x = self.encoder(token_embed, src_key_padding_mask=mask)  # [B, N, D]

        # Average pooling
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            valid_counts = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
        else:
            valid_counts = torch.tensor(N, device=x.device).unsqueeze(0)
        x_pooled = x.sum(dim=1) / valid_counts  # [B, D]

        # Output prediction
        out = self.output_head(x_pooled)  # [B, 1]
        return out.squeeze(-1)

