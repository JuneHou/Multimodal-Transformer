import re
import os
import torch
import operator
from statistics import mean,stdev
import fnmatch

import shutil


def construct_path(args, eval_score=None):
    # Base directory for outputs
    base_dir = os.path.join(args.output_dir, args.task, args.modeltype)
    
    # Additional specifics based on model type and settings
    if 'Bert' in args.modeltype:
        specific_dir = os.path.join(args.model_name, args.notes_order, str(args.num_of_notes), str(args.max_length))
    elif 'TS' in args.modeltype and args.irregular_learn_emb_ts:
        specific_dir = f"TS_{args.tt_max}/{args.TS_model}"
    elif 'Text' in args.modeltype and args.irregular_learn_emb_text:
        specific_dir = f"Text_{args.tt_max}/{args.model_name}/{args.max_length}"
    elif 'CXR' in args.modeltype and args.irregular_learn_emb_cxr:
        specific_dir = f"CXR_{args.tt_max}/{args.model_name}"
    elif 'ECG' in args.modeltype and args.irregular_learn_emb_ecg:
        specific_dir = f"ECG_{args.tt_max}/{args.model_name}"
    else:
        specific_dir = "general"
    
    # Combine into full directory path
    full_dir = os.path.join(base_dir, specific_dir)
    
    # If using specific evaluation score for subdirectories
    if eval_score:
        full_dir = os.path.join(full_dir, eval_score)

    # Ensure directory exists
    os.makedirs(full_dir, exist_ok=True)
    
    # Construct filename
    args.ck_file_path=full_dir
    print(args.ck_file_path)
    
    return full_dir

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"Checkpoint saved at: {path}")

def load_checkpoint(path):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        return torch.load(path, map_location=torch.device('cpu'))
    else:
        print(f"Checkpoint not found at {path}")
        return None

def check_point(all_val, model, all_logits, args, eval_score=None):
    checkpoint_dir = construct_path(args, eval_score)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.seed}.pth.tar")
    
    should_save = False
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        current_best_val = checkpoint['best_val'].get(eval_score, float('-inf'))
        new_val = all_val.get(eval_score, float('inf'))
        should_save = new_val > current_best_val
        print(f"Check for best: Existing {eval_score} = {current_best_val}, New {eval_score} = {new_val}")
    else:
        print(f"No checkpoint exists. Saving initial checkpoint as baseline.")
        should_save = True

    if should_save:
        save_checkpoint({
            'network': model.state_dict(),
            'logits': all_logits,
            'best_val': all_val,
            'args': args
        }, checkpoint_path)
    else:
        print("No new checkpoint saved. Current checkpoint is better or equal.")


if __name__ == "__main__":
    dst='test/'
    copy_file(dst, src=os.getcwd())
