import torch
import torch.nn as nn
import torch.optim as optim
import transformers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb
from core.hme import HierarchicalMoE
from core.sparse_moe import MoE
from utils.config import MoEConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from utils.checkpoint import *


class FeatureProjection(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeatureProjection, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class MULTCrossModel(nn.Module):
    def __init__(self, args, device, orig_d_txt, orig_d_visual, orig_d_acoustic):
        super(MULTCrossModel, self).__init__()
        self.text_projection = FeatureProjection(orig_d_txt, 512, args.embed_dim)
        self.visual_projection = FeatureProjection(orig_d_visual, 512, args.embed_dim)
        self.acoustic_projection = FeatureProjection(orig_d_acoustic, 512, args.embed_dim)

        self.moe = MoE(MoEConfig(
            num_experts=args.num_of_experts[0],
            moe_input_size=args.embed_dim * 3,  # Assuming concatenation of projected features
            moe_hidden_size=args.hidden_size,
            moe_output_size=args.num_labels,
            top_k=args.top_k[0],
            router_type=args.router_type,
            gating=args.gating_function[0],
            num_modalities=args.num_modalities
        ))

    def forward(self, text_features, visual_features, acoustic_features):
        text_features = self.text_projection(text_features)
        visual_features = self.visual_projection(visual_features)
        acoustic_features = self.acoustic_projection(acoustic_features)

        if self.moe.router_type == 'joint':
            combined_features = torch.cat([text_features, visual_features, acoustic_features], dim=1)
            output = self.moe(combined_features)
        elif self.moe.router_type == 'permod':
            output = self.moe([text_features, visual_features, acoustic_features])
        return output
    

def trainer_irg(model, args, accelerator, train_dataloader, dev_dataloader, test_dataloader, device, optimizer, writer=None):
    # Define the loss function
    loss_function = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.num_train_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Prepare inputs and labels
            text_features = batch['text_features'].to(device)
            visual_features = batch['visual_features'].to(device)
            acoustic_features = batch['acoustic_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, loss = model(text_features, visual_features, acoustic_features)
            # Now 'outputs' only contains the tensor 'y', which can be used directly in the loss function

            # Accelerate handles the backward pass
            accelerator.backward(loss)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        # Log training loss
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}, Training Loss: {avg_train_loss}")
        if writer:
            writer.add_scalar('Training Loss', avg_train_loss, epoch)

        # Evaluate the model on validation data
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in dev_dataloader:
            with torch.no_grad():
                text_features = batch['text_features'].to(device)
                visual_features = batch['visual_features'].to(device)
                acoustic_features = batch['acoustic_features'].to(device)
                labels = batch['labels'].to(device)
                
                outputs, loss = model(text_features, visual_features, acoustic_features)
                
                _, predicted = torch.max(outputs, 1)
                total_eval_accuracy += (predicted == labels).sum().item()
                total_eval_loss += loss.item()
        
        avg_val_accuracy = total_eval_accuracy / len(dev_dataloader.dataset)
        avg_val_loss = total_eval_loss / len(dev_dataloader)
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}, Validation Loss: {avg_val_loss}, Accuracy: {avg_val_accuracy * 100:.2f}%")
        if writer:
            writer.add_scalar('Validation Loss', avg_val_loss, epoch)
            writer.add_scalar('Validation Accuracy', avg_val_accuracy, epoch)

        # # Save the model at the end of each epoch or based on your save criteria
        # if args.output_dir:
        #     save_checkpoint(model, optimizer, epoch, args.output_dir)