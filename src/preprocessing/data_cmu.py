import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, SequentialSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import logging

logging.basicConfig(level=logging.DEBUG)

def collate_fn(batch):
    """
    Collate function to process and combine items into a batch.
    """
    text_features = pad_sequence([item['text_features'] for item in batch], batch_first=True, padding_value=0)
    visual_features = pad_sequence([item['visual_features'] for item in batch], batch_first=True, padding_value=0)
    acoustic_features = pad_sequence([item['acoustic_features'] for item in batch], batch_first=True, padding_value=0)
    labels = torch.stack([item['label'] for item in batch])
    ids = [item['ids'] for item in batch]

    # print("text shape:", text_features.shape)
    # print("visual shape:", visual_features.shape)
    # print("acoustic shape:", acoustic_features.shape)
    # print("labels shape:", labels.shape)
    # print("ids shape:", len(ids))

    return {
        'text_features': text_features,
        'visual_features': visual_features,
        'acoustic_features': acoustic_features,
        'labels': labels,
        'ids': ids
    }

def data_prepare(args, mode, data=None):
    """
    Prepare the data for training or evaluation.

    Args:
        args (object): The arguments object.
        mode (str): The mode, either 'train' or 'eval'.
        tokenizer (object): The tokenizer object.
        data (list, optional): The data to be used. Defaults to None.

    Returns:
        dataset (object): The dataset object.
        dataloader (object): The dataloader object.
    """
    dataset = T5_CMUData(args.file_path, mode)

    if mode == 'train':
        dataloader = DataLoader(dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn, drop_last=True)

    return dataset, dataloader

class T5_CMUData(Dataset):
    def __init__(self, file_path, mode, debug=False):
        """
        Initializes the dataset.
        """
        self.data = self.load_data(os.path.join(file_path, f"{mode}.pkl"), debug)
        print(f"Loaded {mode} data with {len(self.data)} samples.")

    def __getitem__(self, idx):
        """
        Retrieves the dataset item at index `idx`.
        """
        sample = self.data[idx]
        text_features = torch.tensor(sample['text_features'], dtype=torch.float32)
        visual_features = torch.tensor(sample['visual_features'], dtype=torch.float32)
        acoustic_features = torch.tensor(sample['acoustic_features'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        segment_id = sample['segment_id']

        return {
            'text_features': text_features,
            'visual_features': visual_features,
            'acoustic_features': acoustic_features,
            'label': label,
            'ids': segment_id
        }

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path, debug):
        """
        Loads data from a pickle file.
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data