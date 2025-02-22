import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        # Normalize embeddings to unit length
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        z = torch.cat([z_i, z_j], dim=0)
        N = z_i.size(0)
        sim = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim.fill_diagonal_(-float('inf'))  # Mask self-comparison
        cos_sim = torch.nn.functional.cosine_similarity(z_i[0:1], z_j[0:1], dim=1)
        print("Manual cosine similarity check:", cos_sim)

        labels = torch.arange(N, dtype=torch.long, device=self.device)
        labels = torch.cat([labels, labels], dim=0)

        # Cross-Entropy Loss calculation
        logits = sim
        loss = F.cross_entropy(logits, labels)

        return loss

