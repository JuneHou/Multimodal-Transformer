import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int):
        super(MLP, self).__init__()
        # input size = 6144 = 48*128
        self.fc1 = nn.Linear(input_size, hidden_size)
        # output size = batch_size, every sample has own label
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class SimCLR(nn.Module):
    def __init__(self, num_mod=4, tt_max=48, batch_size=8, embed_dim=128, hidden_dim=4096, 
                    output_dim=128, temperature=0.05, device='cuda'):
        super(SimCLR, self).__init__()
        self.device = device
        self.num_mod = num_mod
        self.tt_max = tt_max
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.proj_head = MLP(embed_dim*tt_max, output_dim, hidden_dim).to(self.device)

    def forward(self, x):
        # x: [proj_ts_x, proj_txt_x, proj_cxr_x, proj_ecg_x], each of shape [tt_max, batch_size, embed_dim]

        for i in range(self.num_mod):
            x[i] = torch.permute(x[i], (1, 0, 2))
            x[i] = x[i].reshape(self.batch_size, self.tt_max*self.embed_dim)

        x = torch.cat(x, dim=0)
        x = self.proj_head(x) # [batch_size*num_mod, output_dim]

        # https://github.com/sthalles/SimCLR/blob/master/simclr.py
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.num_mod)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        x = F.normalize(x, dim=1)

        similarity_matrix = torch.matmul(x, x.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature

        loss = self.criterion(logits, labels)
        return loss






class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, num_mod=4, tt_max=48, batch_size=2, embed_dim=128, hidden_dim=4096, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim=2)
        self.proj_head = MLP(embed_dim, 128, hidden_dim)

    def forward(self, z_i, z_j):
        # Normalize embeddings to unit length
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        z = torch.cat([z_i, z_j], dim=0)
        N = z_i.size(0)
        sim = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim.fill_diagonal_(-float('inf'))  # Mask self-comparison

        labels = torch.arange(N, dtype=torch.long, device=self.device)
        labels = torch.cat([labels, labels], dim=0)

        # Cross-Entropy Loss calculation
        logits = sim
        loss = F.cross_entropy(logits, labels)

        print("Normalized z_i: ", z_i)
        print("Normalized z_j: ", z_j)
        print("Similarity matrix: ", sim)
        print("Logits: ", logits)


        return loss

