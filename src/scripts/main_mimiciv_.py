import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import warnings
import time
import sys
import logging
import os
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_ import *
from core.train_ import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
from core.interp import *


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def main():
    args = parse_args()
    print(args)

    # Handling mixed precision setup
    if args.fp16:
        args.mixed_precision = "fp16"
    else:
        args.mixed_precision = "no"
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)
    device = accelerator.device
    print(device)

    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup TensorBoard if specified
    if args.tensorboard_dir is not None:
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer = None

    # Configure logging
    warnings.filterwarnings('ignore')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set random seeds for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Update and use the new directory path
    model_dir = construct_path(args)  # This function will handle directory creation inside it
    print(f"Model and checkpoint directory set at: {model_dir}")

    # Optionally, copy model files to the checkpoint directory if seed is 0
    if args.seed == 0:
        model_subdir = os.path.join(model_dir, 'model')
        os.makedirs(model_subdir, exist_ok=True)
        copy_file(model_subdir, src=os.getcwd())
        
    if args.mode=='train':
        if 'Text' in args.modeltype:
            BioBert, BioBertConfig, tokenizer = loadBert(args,device)
        else:
            BioBert, tokenizer = None, None

        from preprocessing.data_mimiciv_ import data_perpare

        train_dataset, train_dataloader = data_perpare(args, 'train', tokenizer)
        val_dataset, val_dataloader = data_perpare(args, 'val', tokenizer)
        _, test_data_loader = data_perpare(args,'test',tokenizer)

    if args.modeltype == 'Text':
        # pure text
        #model= TextModel(args=args,device=device,orig_d_txt=768,Biobert=BioBert)
        model= TextMoE(args=args,device=device,orig_d_txt=768,text_seq_num=args.num_of_notes,Biobert=BioBert)
    elif args.modeltype == 'CXR':
        model = CXRMoE(args=args,device=device,orig_d_cxr=1024)
    elif args.modeltype == 'ECG':
        model = ECGMoE(args=args,device=device,orig_d_ecg=256)
    elif args.modeltype == 'TS':
        # pure time series
        model= TSMixed(args=args,device=device,orig_d_ts=30,orig_reg_d_ts=60, ts_seq_num=args.tt_max)
    else:
        # multimodal fusion
        model= MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert )
    print(device)
    
    # TODO: CXR learning rate
    if args.modeltype=='TS':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif args.modeltype=='TS_CXR':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif args.modeltype=='TS_ECG':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif 'Text' in args.modeltype:
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
    elif "CXR" in args.modeltype:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.txt_learning_rate)
    elif "ECG" in args.modeltype:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    else:
        raise ValueError("Unknown modeltype in optimizer.")

    model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
    accelerator.prepare(model, optimizer, train_dataloader,val_dataloader,test_data_loader)

    trainer_irg(model=model,args=args,accelerator=accelerator,train_dataloader=train_dataloader,\
        dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device,\
        optimizer=optimizer,writer=writer)
    eval_test(args,model,test_data_loader, device, mode='test')
    eval_test(args,model,train_dataloader, device, mode='train')
    eval_test(args,model,val_dataloader, device, mode='val')
    print(f"New maximum memory allocated on GPU: {torch.cuda.max_memory_allocated(device)} bytes")
    print(f'Results saved in:\n{args.ck_file_path}')

    # attention_weights = model.aggregate_attention_weights()
    # pickle.dump(attention_weights, open(os.path.join(args.ck_file_path, "final_attention_weights.pkl"), "wb"))


if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
