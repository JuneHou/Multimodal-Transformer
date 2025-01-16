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
    num_rounds=3

    for round in range(num_rounds):

        for modeltype in args.modeltype:
            print("Model type: ", modeltype)
            mod_count = 0
            if "Text" in modeltype:
                mod_count += 1
            if "CXR" in modeltype:
                mod_count += 1
            if "ECG" in modeltype:
                mod_count += 1
            if "TS" in modeltype:
                mod_count += 1
            args.num_modalities=mod_count

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
            model_dir = construct_path(args, modeltype=modeltype)  # This function will handle directory creation inside it
            print(f"Model and checkpoint directory set at: {model_dir}")

            # Optionally, copy model files to the checkpoint directory if seed is 0
            if args.seed == 0:
                model_subdir = os.path.join(model_dir, 'model')
                os.makedirs(model_subdir, exist_ok=True)
                copy_file(model_subdir, src=os.getcwd())
                    
            if args.mode=='train':
                if 'Text' in modeltype:
                    BioBert, BioBertConfig, tokenizer = loadBert(args,device)
                else:
                    BioBert, tokenizer = None, None

                from preprocessing.data_mimiciv_ import data_perpare

                train_dataset, train_dataloader = data_perpare(args, modeltype, 'train', tokenizer)
                val_dataset, val_dataloader = data_perpare(args, modeltype, 'val', tokenizer)
                _, test_data_loader = data_perpare(args, modeltype,'test',tokenizer)

            if modeltype == 'Text':
                # pure text
                #model= TextModel(args=args,device=device,orig_d_txt=768,Biobert=BioBert)
                model= TextMoE(args=args,device=device,modeltype=modeltype,orig_d_txt=768,text_seq_num=args.num_of_notes,Biobert=BioBert)
            elif modeltype == 'CXR':
                model = CXRMoE(args=args,device=device,modeltype=modeltype,orig_d_cxr=1024)
            elif modeltype == 'ECG':
                model = ECGMoE(args=args,device=device,modeltype=modeltype,orig_d_ecg=256)
            elif modeltype == 'TS':
                # pure time series
                model= TSMixed(args=args,device=device,modeltype=modeltype,orig_d_ts=30,orig_reg_d_ts=60, ts_seq_num=args.tt_max)
            else:
                # multimodal fusion
                model= MULTCrossModel(args=args,device=device,modeltype=modeltype,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert )
            print(device)
            
            # TODO: CXR learning rate
            if modeltype=='TS':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
            elif modeltype=='TS_CXR':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
            elif modeltype=='TS_ECG':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
            elif 'Text' in modeltype:
                optimizer= torch.optim.Adam([
                        {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
                    ], lr=args.ts_learning_rate)
            elif "CXR" in modeltype:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.txt_learning_rate)
            elif "ECG" in modeltype:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
            else:
                raise ValueError("Unknown modeltype in optimizer.")

            model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
            accelerator.prepare(model, optimizer, train_dataloader,val_dataloader,test_data_loader)

            trainer_irg(model=model,args=args,accelerator=accelerator,modeltype=modeltype,train_dataloader=train_dataloader,\
                dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device,\
                optimizer=optimizer,writer=writer)
            eval_test(args,modeltype,model,test_data_loader, device, mode='test')
            eval_test(args,modeltype,model,train_dataloader, device, mode='train')
            eval_test(args,modeltype,model,val_dataloader, device, mode='val')
            print(f"New maximum memory allocated on GPU: {torch.cuda.max_memory_allocated(device)} bytes")
            print(f'Results saved in:\n{args.ck_file_path}')

        datasets = ['train', 'val', 'test']
        # Example usage: train_los-48-cxr-notes-ecg_stays.pkl
        for dataset in datasets:
            print(f"Starting los {dataset} dataset")
            ts_pred = pd.read_csv(f'{args.output_dir}/TS_{dataset}_results.csv')
            print("number of ts_pred: ", len(ts_pred))
            text_pred = pd.read_csv(f'{args.output_dir}/Text_{dataset}_results.csv')
            print("number of text_pred: ", len(text_pred))
            cxr_pred = pd.read_csv(f'{args.output_dir}/CXR_{dataset}_results.csv')
            print("number of cxr_pred: ", len(cxr_pred))
            ecg_pred = pd.read_csv(f'{args.output_dir}/ECG_{dataset}_results.csv')
            print("number of ecg_pred: ", len(ecg_pred))
            multi_pred = pd.read_csv(f'{args.output_dir}/TS_CXR_Text_ECG_{dataset}_results.csv')
            print("number of multi_pred: ", len(multi_pred))

            kl_scores = assign_pid_4probs(ts_pred, text_pred, cxr_pred, ecg_pred, multi_pred)
            file_path = f'{args.file_path}/{dataset}_los-48-cxr-notes-ecg-missingInd_stays.pkl'
            new_stays_list = update_stays_with_weights(file_path, kl_scores)

if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
