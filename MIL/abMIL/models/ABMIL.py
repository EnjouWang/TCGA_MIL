import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_attention_entropy(attn, eps=1e-12):
    """
    Normalized entropy in [0, 1]
    """
    if attn.dim() == 2:
        attn = attn.squeeze(0)

    N = attn.numel()
    attn = attn.clamp(min=eps)

    entropy = -torch.sum(attn * torch.log(attn))
    entropy_norm = entropy / torch.log(torch.tensor(N, device=attn.device, dtype=attn.dtype))
    return entropy_norm


class ABMILPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim)
        self.U = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, feats):
        """
        feats: [B, L, D]
        mask:  [B, L] (1 = valid, 0 = padded)
        return: pooled [B, D]
        """

        # attention score before mask: [B, L, 1]
        H = torch.tanh(self.V(feats)) * torch.sigmoid(self.U(feats))
        A = self.w(H).squeeze(-1)  # [B, L]

        # softmax over L dimension
        A = torch.softmax(A, dim=1)  # [B, L]
        # print(normalized_attention_entropy(A, eps=1e-12).item())
        # print(A.shape)
        # print(f"group featuere{A[0, -5:]}")
        # print(f"patch featuere{A[0, :5]}")
        # print(torch.mean(A))

        # weighted sum
        pooled = torch.bmm(A.unsqueeze(1), feats).squeeze(1)  # [B, D]
        return pooled, A
