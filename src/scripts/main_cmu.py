import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from accelerate import Accelerator
from argparse import ArgumentParser

# Assuming relative imports based on your project structure
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from core.model import MULTCrossModel
#from core.train import trainer_irg, eval_test
from preprocessing.data_cmu import data_prepare
#from moe_cmu import *
from core.model_cmu import MULTCrossModel
from core.train_cmu import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
#from core.interp import *

def parse_args():
    parser = ArgumentParser(description="Train and Evaluate the Multimodal Model")
    parser.add_argument('--file_path', type=str, default='./data', help='Path to the data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory for output')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs', help='Directory for TensorBoard logs')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 training')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU instead of GPU')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='Mode of operation')
    parser.add_argument('--cross_method', type=str, default='moe', choices=['moe','MulT', 'MAGGate', 'Outer'], help='Cross method')
    parser.add_argument('--num_modalities', type=int, default=3, help='Number of modalities')
    parser.add_argument('--train_batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--num_train_epochs', type=int, default=8, help='Number of epochs to train the model')
    parser.add_argument('--kernel_size', type=int, default=1, help='Kernel size for the convolutional layers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Number of gradient accumulation steps')
    parser.add_argument('--layers', type=int, default=3, help='Number of layers in the model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads in the model')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension for the model')
    parser.add_argument("--dropout", default=0.10, type=float, help="dropout.")
    parser.add_argument('--task', type=str, default='multiclass', choices=['multiclass', 'multilabel'], help='Type of task')
    parser.add_argument('--num_labels', type=int, default=3, help='Number of labels in the dataset')
    parser.add_argument("--num_of_experts", nargs='*', type=int, help="number of MLPs in MoE, for HME need to specify each level")
    parser.add_argument("--top_k", nargs='*', type=int, help="top k experts to select")
    parser.add_argument("--router_type", type=str, default='permod', help='Type of router')
    parser.add_argument("--hidden_size", type=int, default=512, help='Hidden size for the router')
    parser.add_argument("--gating_function", nargs='*', type=str, help="all gating functions: softmax, laplace, gaussian, enter at least one")
    parser.add_argument("--device", type=str, default='cuda:7', help='Device to use for training')
    parser.add_argument("--tt_max", type=int, default=64, help='Max number of tokens')
    parser.add_argument("--modeltype", type=str, default='Text_Visual_Acoustic', help='Model type')
    parser.add_argument("--debug", action='store_true', help='Debug mode')
    # Add other necessary arguments as needed
    return parser.parse_args()

def main():
    args = parse_args()

    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)
    device = accelerator.device
    print("Using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    if args.seed:
        set_seed(args.seed)

    # Data loading
    train_dataset, train_dataloader = data_prepare(args, 'train')
    val_dataset, val_dataloader = data_prepare(args, 'val')
    _, test_dataloader = data_prepare(args, 'test')

    # Model initialization
    model = MULTCrossModel(args=args, device=device, modeltype="Text_Visual_Acoustic", orig_d_txt=1024, orig_d_visual=47, orig_d_acoustic=74)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model, optimizer, train_dataloader, val_dataloader, test_dataloader = \
        accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_dataloader)

    # Training and evaluation logic
    trainer_irg(model=model, args=args, accelerator=accelerator, train_dataloader=train_dataloader,
                dev_dataloader=val_dataloader, test_dataloader=test_dataloader, device=device,
                optimizer=optimizer)

    print(f"New maximum memory allocated on GPU: {torch.cuda.max_memory_allocated(device)} bytes")
    print(f'Results saved in: {args.output_dir}')

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
