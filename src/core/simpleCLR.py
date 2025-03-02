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

    def forward(self, x, missings):
        # x: [proj_ts_x, proj_txt_x, proj_cxr_x, proj_ecg_x], each of shape [tt_max, batch_size, embed_dim]

        for i in range(self.num_mod):
            x[i] = torch.permute(x[i], (1, 0, 2))
            x[i] = x[i].reshape(self.batch_size, self.tt_max*self.embed_dim)

        x = torch.cat(x, dim=0)
        if torch.any(torch.isnan(x)):
            print("Loss is NaN")
        x = self.proj_head(x) # [batch_size*num_mod, output_dim]

        norm_loss = 0
        # soft norm
        # norm_loss = 0.01 * F.relu(x.norm(dim=1) - 2.0).mean()
        # norm
        # norm_loss = 0.001 * x.norm(dim=1, p=2).mean()


        # https://github.com/sthalles/SimCLR/blob/master/simclr.py
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.num_mod)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        if torch.any(torch.isnan(x)):
            print("Loss is NaN")
        
        missing_mask = torch.zeros(self.batch_size*self.num_mod, dtype=torch.bool).to(self.device)
        for i, missing in enumerate(missings):
            missing_mask[(i+1)*self.batch_size:(i+2)*self.batch_size] = missing
        x = x * (~missing_mask).unsqueeze(1) + missing_mask.unsqueeze(1)*1e-7
        
        x = F.normalize(x, dim=1)

        if torch.any(torch.isnan(x)):
            print("Loss is NaN")

        similarity_matrix = torch.matmul(x, x.T)

        # discard the main diagonal from both: labels and similarities matrix
        # Mask examples, assume batch_size = 4, num_mod = 2, mod 2 of sample 2 is missing from batch
        # Reglar mask:         Mask with missing modality:
        # |1 0 0 0 0 0 0 0|    |1 0 0 0 0 0 0 0|
        # |0 1 0 0 0 0 0 0|    |0 1 0 0 0 0 0 0|
        # |0 0 1 0 0 0 0 0|    |0 0 1 0 0 0 0 0|
        # |0 0 0 1 0 0 0 0|    |1 1 1 1 1 1 1 1|
        # |0 0 0 0 1 0 0 0|    |0 0 0 0 1 0 0 0|
        # |0 0 0 0 0 1 0 0|    |0 0 0 0 0 1 0 0|
        # |0 0 0 0 0 0 1 0|    |0 0 0 0 0 0 1 0|
        # |0 0 0 0 0 0 0 1|    |0 0 0 0 0 0 0 1|

        # Label equality mask
        # |1 0 0 0 1 0 0 0|
        # |0 1 0 0 0 1 0 0|
        # |0 0 1 0 0 0 1 0|
        # |0 0 0 1 0 0 0 1|
        # |1 0 0 0 1 0 0 0|
        # |0 1 0 0 0 1 0 0|
        # |0 0 1 0 0 0 1 0|
        # |0 0 0 1 0 0 0 1|

        # Mask examples, assume batch_size = 2, num_mod = 4, mod 2 of sample 2 is missing from batch
        # Reglar mask:         Mask with missing modality:
        # |1 0 0 0 0 0 0 0|    |1 0 0 0 0 0 0 0|
        # |0 1 0 0 0 0 0 0|    |0 1 0 0 0 0 0 0|
        # |0 0 1 0 0 0 0 0|    |0 0 1 0 0 0 0 0|
        # |0 0 0 1 0 0 0 0|    |0 0 0 1 0 0 0 0|
        # |0 0 0 0 1 0 0 0|    |0 0 0 0 1 0 0 0|
        # |0 0 0 0 0 1 0 0|    |1 1 1 1 1 1 1 1|
        # |0 0 0 0 0 0 1 0|    |0 0 0 0 0 0 1 0|
        # |0 0 0 0 0 0 0 1|    |0 0 0 0 0 0 0 1|

        # Label equality mask
        # |1 0 1 0 1 0 1 0|
        # |0 1 0 1 0 1 0 1|
        # |1 0 1 0 1 0 1 0|
        # |0 1 0 1 0 1 0 1|
        # |1 0 1 0 1 0 1 0|
        # |0 1 0 1 0 1 0 1|
        # |1 0 1 0 1 0 1 0|
        # |0 1 0 1 0 1 0 1|

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        # add missing modality mask
        # missing_mask = torch.zeros(self.batch_size*self.num_mod, dtype=torch.bool).to(self.device)
        # for i, missing in enumerate(missings):
        #     missing_mask[(i+1)*self.batch_size:(i+2)*self.batch_size] = missing
        mask = mask + missing_mask.unsqueeze(1)

        total_missing = torch.sum(missing_mask)

        labels = labels[~mask].view(labels.shape[0] - total_missing, -1)
        # similarity_matrix = similarity_matrix*(~missing_mask) + missing_mask*1e-7
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0] - total_missing, -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits[8:]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # logits = logits + torch.ones_like(logits).to(self.device) * 1e-7

        logits = logits / self.temperature

        loss = self.criterion(logits, labels)
        if torch.isnan(loss):
            print("Loss is NaN")
        return loss + norm_loss



class SimCLR_pair(nn.Module):
    def __init__(self, num_mod=4, tt_max=48, batch_size=8, embed_dim=128, hidden_dim=4096, 
                 output_dim=128, temperature=0.05, device='cuda'):
        super(SimCLR_pair, self).__init__()
        self.device = device
        self.num_mod = num_mod  # Total number of modalities including TS
        self.tt_max = tt_max
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.proj_head = MLP(embed_dim * tt_max, output_dim, hidden_dim).to(self.device)

    def forward(self, embeddings, txt_missing, cxr_missing, ecg_missing):
        # embeddings: list of tensors for each modality, each tensor of shape [batch_size, tt_max, embed_dim]
        # Each tensor is projected, e.g., proj_x_ts for TS
        # missing flags for each modality except TS which is always present
        
        ts_embeddings = embeddings[0]  # TS embeddings are always present
        ts_embeddings = ts_embeddings.reshape(self.batch_size, self.tt_max * self.embed_dim)
        ts_embeddings = self.proj_head(ts_embeddings)
        ts_embeddings = F.normalize(ts_embeddings, dim=1)

        losses = []
        presence_flags = [not txt_missing, not cxr_missing, not ecg_missing]  # Convert missing flags to presence flags

        for i, present in enumerate(presence_flags, start=1):  # start=1 to skip TS
            if present and i < len(embeddings):
                other_embeddings = embeddings[i]
                other_embeddings = other_embeddings.reshape(self.batch_size, self.tt_max * self.embed_dim)
                other_embeddings = self.proj_head(other_embeddings)
                other_embeddings = F.normalize(other_embeddings, dim=1)

                # Compute similarity matrix for the current modality pair (TS and other)
                similarity_matrix = torch.matmul(ts_embeddings, other_embeddings.T) / self.temperature

                # Create labels for the positive pairs
                labels = torch.arange(self.batch_size, device=self.device)
                loss = self.criterion(similarity_matrix, labels)
                losses.append(loss)

        # Normalize the total loss by the number of valid modality pairs computed
        total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=self.device)

        return total_loss


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

