# import argparse
from utils.util import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import pdb


def data_perpare(args,modeltype,mode,tokenizer,data=None):
    """
    Prepare the data for training or evaluation.

    Args:
        args (object): The arguments object.
        mode (str): The mode, either 'train' or 'eval'.
        tokenizer (object): The tokenizer object.
        data (list, optional): The data to be used. Defaults to None.

    Returns:
        dataset (object): The dataset object.
        sampler (object): The sampler object.
        dataloader (object): The dataloader object.
    """
    dataset=TSNote_Irg(args,modeltype, mode, tokenizer, data=data)

    if mode=='train':
        #sampler = RandomSampler(dataset)
        dataloader= DataLoader(dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=TextTSIrgcollate_fn, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
        dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=TextTSIrgcollate_fn, drop_last=True)

    return dataset, dataloader


def F_impute(X,tt,mask,duration,tt_max):
    """
    Imputes missing values in the input data based on the discretization rule mentioned in the paper.

    Parameters:
    X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
    tt (numpy.ndarray): Array of time values corresponding to each sample.
    mask (numpy.ndarray): Array indicating missing values in the input data.
    duration (int): Duration of each time interval for discretization.
    tt_max (int): Maximum time value.

    Returns:
    numpy.ndarray: Imputed data matrix of shape (tt_max//duration, n_features*2).
    """
    
    no_feature=X.shape[1]
    impute=np.zeros(shape=(tt_max//duration,no_feature*2))
    for  x,t,m in zip(X,tt,mask):
        row=int(t/duration)
        if row>=tt_max:
            continue
        for  f_idx, (rwo_x, row_m) in enumerate(zip(x,m)):
            # perform imputation according to the discretization rule in paper
            if row_m==1:
                impute[row][no_feature+f_idx]=1
                impute[row][f_idx]=rwo_x
            else:
                if impute[row-1][f_idx]!=0:
                    impute[row][f_idx]=impute[row-1][f_idx]

    return impute


class TSNote_Irg(Dataset):
    """
    A PyTorch dataset class for handling time series note data in the MIMIC-IV dataset.

    Args:
        args (argparse.Namespace): The command-line arguments.
        mode (str): The mode of the dataset (e.g., "train", "val", "test").
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the text data.
        data (list, optional): The list of data samples. If not provided, the data will be loaded from a file.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the text data.
        max_len (int): The maximum length of the input sequences.
        data (list): The list of data samples.
        chunk (bool): Whether to chunk the data.
        text_id_attn_data (list): The list of text data samples for attention calculation.
        padding (str): The padding strategy for the input sequences.
        notes_order (str): The order of the notes.
        order_sample (numpy.ndarray): The array of randomly sampled note orders.
        modeltype (str): The type of the model.
        model_name (str): The name of the model.
        num_of_notes (int): The number of notes to consider.
        tt_max (float): The maximum value of the time-to-end feature.

    Methods:
        __getitem__(self, idx): Retrieves the data at the given index.
        __len__(self): Returns the length of the dataset.
    """
    
    def __init__(self,args,modeltype,mode,tokenizer,data=None):
        self.tokenizer = tokenizer
        self.max_len = args.max_length
        print(args.file_path)
        if data != None:
            self.data=data
        else:
            self.data = load_data(file_path=args.file_path,mode=mode,debug=args.debug,task=args.task)
        self.chunk=args.chunk
        if self.chunk:
            self.text_id_attn_data = load_data(file_path=args.file_path,mode=mode,text=True,task=args.task)
        self.padding= "max_length" if args.pad_to_max_length  else False

        if mode=="train":
            self.notes_order=args.notes_order
        else:
            self.notes_order="Last"

        if args.ratio_notes_order!=None:
            self.order_sample=np.random.binomial(1, args.ratio_notes_order,len(self.data))

        self.modeltype=modeltype
        self.model_name=args.model_name
        self.num_of_notes=args.num_of_notes
        self.tt_max=args.tt_max
        self.reg_ts = args.reg_ts
        self.use_pt_text_embeddings=args.use_pt_text_embeddings
        
    def __getitem__(self, idx):
        """
        Retrieves the data at the given index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            dict: A dictionary containing the data at the given index.
        """
        if self.notes_order!=None:
            notes_order=self.notes_order
        else:
            # notes_order= 'Last' if self.order_sample[idx]==1  else 'First'
            notes_order = 'Last'
        
        data_detail = self.data[idx]
        idx=data_detail['name']
        ids = str(round(data_detail.get("hadm_id"))) + str(data_detail.get("stay_id"))

        text_token=[]
        atten_mask=[]
        label=data_detail["label"]

        if "TS" in self.modeltype:
            ts_tt=data_detail["ts_tt"]
            reg_ts=data_detail['reg_ts']
            ts=data_detail['irg_ts']
            ts_mask=data_detail['irg_ts_mask']
            ts_tt=torch.tensor([t/self.tt_max for t in ts_tt],dtype=torch.float)
            reg_ts=F_impute(ts,ts_tt,ts_mask,1,self.tt_max)
            reg_ts=torch.tensor(reg_ts,dtype=torch.float)
            ts=torch.tensor(ts,dtype=torch.float)
            ts_mask=torch.tensor(ts_mask,dtype=torch.long)
        else:
            reg_ts=None
            ts=None
            ts_mask=None
            ts_tt=None
        
        if 'text_data' not in data_detail.keys():
            text = ""
        else:
            text = data_detail['text_data']
            text_time_to_end=data_detail["text_time_to_end"]

        # if len(text)==0:
        #     return Non

        # reg_ts = data_detail['reg_ts']

        if 'Text' in self.modeltype and not data_detail['text_missing']:
            text_emb = data_detail['text_embeddings']
            text_emb = torch.tensor(text_emb, dtype=torch.float)

            text_time_to_end=[1-t/self.tt_max for t in text_time_to_end]
            text_time_mask=[1]*len(text_time_to_end)

            for t in text:
                inputs = self.tokenizer.encode_plus(t, padding=self.padding,\
                                                    max_length=self.max_len,\
                                                    add_special_tokens=True,\
                                                    return_attention_mask = True,\
                                                    truncation=True)
                text_token.append(torch.tensor(inputs['input_ids'],dtype=torch.long))
                attention_mask=inputs['attention_mask']
                if "Longformer" in self.model_name:

                    attention_mask[0]+=1
                    atten_mask.append(torch.tensor(attention_mask,dtype=torch.long))
                else:
                    atten_mask.append(torch.tensor(attention_mask,dtype=torch.long))

            while len(text_token)<self.num_of_notes:
                text_token.append(torch.tensor([0],dtype=torch.long))
                atten_mask.append(torch.tensor([0],dtype=torch.long))
                
                if not self.use_pt_text_embeddings:
                    text_time_to_end.append(0)
                    text_time_mask.append(0)

            text_time_to_end=torch.tensor(text_time_to_end,dtype=torch.float)
            text_time_mask=torch.tensor(text_time_mask,dtype=torch.long)

            if notes_order == "Last":
                text_emb=text_emb[-self.num_of_notes:]
                text_token = text_token[-self.num_of_notes:]
                atten_mask = atten_mask[-self.num_of_notes:]
                text_time_to_end = text_time_to_end[-self.num_of_notes:]
                text_time_mask = text_time_mask[-self.num_of_notes:]
            else:
                text_emb=text_emb[:self.num_of_notes]
                text_token = text_token[:self.num_of_notes]
                atten_mask = atten_mask[:self.num_of_notes]
                text_time_to_end = text_time_to_end[:self.num_of_notes]
                text_time_mask = text_time_mask[:self.num_of_notes]
        else:
            text_token = [torch.zeros(100) for _ in range(self.num_of_notes)]
            atten_mask = [torch.zeros(100) for _ in range(self.num_of_notes)]
            text_emb = [torch.zeros(768)]
            text_time_to_end = torch.zeros(1)
            text_time_mask = torch.ones(1)

        if 'CXR' in self.modeltype and not data_detail['cxr_missing']:
            cxr_feats = data_detail['cxr_feats']
            cxr_feats = torch.tensor(cxr_feats, dtype=torch.float)

            cxr_time_to_end = data_detail['cxr_time'].astype(np.float32)
            cxr_time_to_end = torch.tensor(cxr_time_to_end, dtype=torch.float)

            cxr_time_mask = [1] * len(cxr_time_to_end)
            cxr_time_mask = torch.tensor(cxr_time_mask, dtype=torch.long)
        else:
            cxr_feats = torch.zeros((5, 1024))
            cxr_time_to_end = torch.zeros(5)
            cxr_time_mask = torch.ones(5)

        if 'ECG' in self.modeltype and not data_detail['ecg_missing']:
            ecg_feats = data_detail['ecg_feats']
            ecg_feats = torch.tensor(ecg_feats, dtype=torch.float)

            # If any ecg_feats are nan, replace with 0
            ecg_feats[torch.isnan(ecg_feats)] = 0

            # If any ecg_feats are inf, replace with 0
            ecg_feats[torch.isinf(ecg_feats)] = 0

            ecg_time_to_end = data_detail['ecg_time'].astype(np.float32)
            ecg_time_to_end = torch.tensor(ecg_time_to_end, dtype=torch.float)

            ecg_time_mask = [1] * len(ecg_time_to_end)
            ecg_time_mask = torch.tensor(ecg_time_mask, dtype=torch.long)
        else:
            ecg_feats = torch.zeros((5, 256))
            ecg_time_to_end = torch.zeros(5)
            ecg_time_mask = torch.ones(5)

        label=torch.tensor(label,dtype=torch.long)
        
        # 'ts_weight', 'text_weight', 'cxr_weight', 'ecg_weight'

        if 'TS_CXR_Text_ECG' in self.modeltype:
            ts_weight = torch.tensor(data_detail['ts_weight'], dtype=torch.float)
            text_weight = torch.tensor(data_detail['text_weight'], dtype=torch.float)
            cxr_weight = torch.tensor(data_detail['cxr_weight'], dtype=torch.float)
            ecg_weight = torch.tensor(data_detail['ecg_weight'], dtype=torch.float)

            data_items = {'ids': ids, 'idx':idx, 'label':label, 
                'ts_weight': ts_weight, 'text_weight': text_weight, 
                'cxr_weight': cxr_weight, 'ecg_weight': ecg_weight}
        else:
            data_items = {'ids': ids, 'idx':idx, 'label':label, 
                'ts_weight':torch.ones(1), 'text_weight':torch.ones(1), 'cxr_weight':torch.ones(1), 'ecg_weight':torch.ones(1)}

        if 'TS' in self.modeltype:
            data_items.update({'ts': ts, 'ts_mask': ts_mask, 'ts_tt': ts_tt, 'reg_ts': reg_ts})
        if 'Text' in self.modeltype:
            data_items.update({'input_ids': text_token, 'attention_mask': atten_mask, 
                            'text_embeddings': text_emb, 'note_time': text_time_to_end, 
                            'text_time_mask': text_time_mask})
        if 'CXR' in self.modeltype:
            data_items.update({'cxr_feats': cxr_feats, 'cxr_time': cxr_time_to_end, 'cxr_time_mask': cxr_time_mask})
        if 'ECG' in self.modeltype:
            data_items.update({'ecg_feats': ecg_feats, 'ecg_time': ecg_time_to_end, 'ecg_time_mask': ecg_time_mask})

        return data_items

    def __len__(self):
        return len(self.data)

def load_data(file_path, mode, debug=False, text=False, task='ihm'):
    """
    Load data from a file.

    Args:
        file_path (str): The path to the file.
        mode (str): The mode of the data.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        text (bool, optional): Whether the data is text. Defaults to False.
        task (str, optional): The task of the data. Defaults to 'ihm'.

    Returns:
        data: The loaded data.
    """
    dataPath = os.path.join(file_path, mode + '_' + task + '_stays.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if debug and not text:
                data = data[:100]

    return data

def TextTSIrgcollate_fn(batch):

    batch = list(filter(lambda x: x is not None, batch))
    ids = [example['ids'] for example in batch]
    if 'ts' in batch[0].keys():
        batch = list(filter(lambda x: len(x['ts']) <1000, batch))
    if not batch:
        return None  # Or however you want to handle an empty batch in your loop
    label = torch.stack([example["label"] for example in batch])
    if len(batch) == 0:
        return

    if 'cxr_missing' in batch[0].keys():
        cxr_missing = torch.stack([torch.tensor(example["cxr_missing"]) for example in batch])
    else:
        cxr_missing = None
    if 'text_missing' in batch[0].keys():
        text_missing = torch.stack([torch.tensor(example["text_missing"]) for example in batch])
    else:
        text_missing = None
    if 'ecg_missing' in batch[0].keys():
        ecg_missing = torch.stack([torch.tensor(example["ecg_missing"]) for example in batch])
    else:
        ecg_missing = None

    if 'ts' in batch[0].keys():
        ts_input_sequences = pad_sequence([example['ts'] for example in batch], batch_first=True, padding_value=0)
        ts_mask_sequences = pad_sequence([example['ts_mask'] for example in batch], batch_first=True, padding_value=0)
        ts_tt = pad_sequence([example['ts_tt'] for example in batch], batch_first=True, padding_value=0 )
        
        if batch[0]['reg_ts'] is not None:
            reg_ts_input=torch.stack([example['reg_ts'] for example in batch])
        else:
            reg_ts_input=None
    else:
        ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input = None, None, None, None

    # except:
    #     # if there is no vital signs, just return
    #     return

    if 'input_ids' in batch[0].keys():
        input_ids=[pad_sequence(example['input_ids'],batch_first=True,padding_value=0).transpose(0,1) for example in batch]
        attn_mask=[pad_sequence(example['attention_mask'],batch_first=True,padding_value=0).transpose(0,1) for example in batch]

        input_ids=pad_sequence(input_ids,batch_first=True,padding_value=0).transpose(1,2)
        attn_mask=pad_sequence(attn_mask,batch_first=True,padding_value=0).transpose(1,2)
    else:
        input_ids, attn_mask = None, None
    
    if 'note_time' in batch[0].keys():
        note_time=pad_sequence([torch.tensor(example['note_time'],dtype=torch.float) for example in batch],batch_first=True,padding_value=0)
        note_time_mask=pad_sequence([torch.tensor(example['text_time_mask'],dtype=torch.long) for example in batch],batch_first=True,padding_value=0)
    else:
        note_time, note_time_mask = None, None

    if 'text_embeddings' in batch[0].keys():
        text_emb = [pad_sequence(example['text_embeddings'], batch_first=True, padding_value=0) for example in batch]
        text_emb = pad_sequence(text_emb, batch_first=True, padding_value=0)
    else:
        text_emb = None

    if 'cxr_feats' in batch[0].keys():
        # cxr_feats=pad_sequence([example['cxr_feats'] for example in batch],batch_first=True,padding_value=0 )
        cxr_feats = [pad_sequence(example['cxr_feats'], batch_first=True, padding_value=0) for example in batch]
        cxr_feats = pad_sequence(cxr_feats, batch_first=True, padding_value=0)
        cxr_time = pad_sequence([torch.tensor(example['cxr_time'], dtype=torch.float) for example in batch], batch_first=True, padding_value=0)
        cxr_time_mask = pad_sequence([torch.tensor(example['cxr_time_mask'], dtype=torch.long) for example in batch], batch_first=True, padding_value=0)
    else:
        cxr_feats, cxr_time, cxr_time_mask = None, None, None

    if 'ecg_feats' in batch[0].keys():
        ecg_feats = [pad_sequence(example['ecg_feats'], batch_first=True, padding_value=0) for example in batch]
        ecg_feats = pad_sequence(ecg_feats, batch_first=True, padding_value=0)

        ecg_time = pad_sequence([torch.tensor(example['ecg_time'], dtype=torch.float) for example in batch], batch_first=True, padding_value=0)
        ecg_time_mask = pad_sequence([torch.tensor(example['ecg_time_mask'], dtype=torch.long) for example in batch], batch_first=True, padding_value=0)
    else:
        ecg_feats, ecg_time, ecg_time_mask = None, None, None

    ts_weight = torch.stack([example['ts_weight'] for example in batch])
    text_weight = torch.stack([example['text_weight'] for example in batch])
    cxr_weight = torch.stack([example['cxr_weight'] for example in batch])
    ecg_weight = torch.stack([example['ecg_weight'] for example in batch])

    return ids, ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, \
         input_ids, attn_mask, text_emb, note_time, note_time_mask, cxr_feats, cxr_time, cxr_time_mask, ecg_feats, \
            ecg_time, ecg_time_mask, label, cxr_missing, text_missing, ecg_missing, ts_weight, text_weight, cxr_weight, ecg_weight
