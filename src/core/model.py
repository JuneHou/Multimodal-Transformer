import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from core.module import *
from core.interp import *
import copy
import pdb


class BertForRepresentation(nn.Module):
    """
    This class represents a BERT model for text representation.

    Args:
        args (object): The arguments for the model.
        BioBert (object): The BioBERT model.

    Attributes:
        bert (object): The BioBERT model.
        dropout (object): The dropout layer.
        model_name (str): The name of the model.
    """
    
    def __init__(self, args,BioBert):
        super().__init__()
        self.bert = BioBert

        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name=args.model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None , doc_idx_list=None):
        """
        Forward pass of the model.

        Args:
            input_ids_sequence (List[Tensor]): List of input token IDs for each sequence.
            attention_mask_sequence (List[Tensor]): List of attention masks for each sequence.
            sent_idx_list (List[int], optional): List of sentence indices. Defaults to None.
            doc_idx_list (List[int], optional): List of document indices. Defaults to None.

        Returns:
            Tensor: Text embeddings for each sequence.
        """
        txt_arr = []

        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):

            input_ids = input_ids.long()
            if 'Longformer' in self.model_name:

                attention_mask-=1

                text_embeddings=self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings=self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr=torch.stack(txt_arr)
        return txt_arr


class TextModel(nn.Module):
    def __init__(self,args,device,orig_d_txt=768,Biobert=None):
        """
        Construct a TextModel.
        """
        super(TextModel, self).__init__()

        self.device=device
        self.task=args.task
        # self.agg_type=args.agg_type
        self.out_dropout = args.dropout
        self.orig_d_txt=orig_d_txt
        self.d_txt= args.embed_dim
        self.bertrep=BertForRepresentation(args,Biobert)

        self.proj_txt =nn.Linear(self.orig_d_txt, self.d_txt)

        output_dim = args.num_labels
        self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        self.out_layer = nn.Linear(self.d_txt, output_dim)

        if 'ihm'  or 'los' in self.task:
            self.loss_fct1=CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")


    def forward(self, input_ids_sequences,
                attn_mask_sequences,labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        x_txt=self.bertrep(input_ids_sequences,attn_mask_sequences)
        x_txt=torch.mean(x_txt,dim=1)
        proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(proj_x_txt)), p=self.out_dropout, training=self.training))
        last_hs_proj += proj_x_txt
        output = self.out_layer(last_hs_proj)

        # if 'ihm' or 'los' in self.task:
        #     if labels!=None:
        #         return self.loss_fct1(output, labels)
        #     return torch.nn.functional.softmax(output,dim=-1)[:,1]
        if 'ihm' in self.task:
            if labels is not None:
                # Continue using binary classification approach for IHM
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output, dim=-1)[:, 1]

        elif 'los' in self.task:
            if labels is not None:
                # Use a multiclass classification approach for LOS
                # Assuming output has the correct shape (batch_size, num_classes)
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output, dim=-1)

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class MULTCrossModel(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,orig_d_txt=None,ts_seq_num=None,text_seq_num=None, Biobert=None):
        """
        Construct a MulT Cross model.
        """
        super(MULTCrossModel, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads
        self.args = args
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr=args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg=args.irregular_learn_emb_ecg
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.tt_max=args.tt_max
        self.cross_method=args.cross_method
        self.num_modalities = args.num_modalities
        self.use_pt_text_embeddings = args.use_pt_text_embeddings
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)

        if self.irregular_learn_emb_ts or self.irregular_learn_emb_text:
            self.time_query=torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time - 1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts=orig_d_ts
            self.d_ts=args.embed_dim
            self.ts_seq_num=ts_seq_num

            if self.irregular_learn_emb_ts:
                self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
 
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq_feature':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype: 
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn_text = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        # if self.modeltype == "TS_CXR":
        if "CXR" in self.modeltype:
            self.orig_d_cxr = 1024
            self.d_cxr = args.embed_dim
            self.cxr_seq_num = 5

            if self.irregular_learn_emb_cxr:
                self.time_attn_cxr = multiTimeAttention(1024, self.d_cxr, args.embed_time, 8)
            else:
                self.proj_cxr = nn.Conv1d(self.orig_d_cxr, self.d_cxr, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if "ECG" in self.modeltype:
            self.orig_d_ecg = 256
            self.d_ecg = args.embed_dim
            self.ecg_seq_num = 5

            if self.irregular_learn_emb_ecg:
                self.time_attn_ecg = multiTimeAttention(256, self.d_ecg, args.embed_time, 8)
            else:
                self.proj_ecg = nn.Conv1d(self.orig_d_ecg, self.d_ecg, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
        
        if "Notes" in self.modeltype:
            self.orig_d_note = 768
            self.d_note = args.embed_dim
            self.note_seq_num = 5

            if self.irregular_learn_emb_note:
                self.time_attn_note = multiTimeAttention(768, self.d_note, args.embed_time, 8)
            else:
                self.proj_note = nn.Conv1d(self.orig_d_note, self.d_note, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
        
        output_dim = args.num_labels
        # if self.modeltype=="TS_Text":
        if self.cross_method in ["self_cross", "moe", "hme"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.cross_layers)
            dim = 0
            if "TS" in self.modeltype:
                dim += self.d_ts
            if "Text" in self.modeltype:
                dim += self.d_txt
            if "CXR" in self.modeltype:
                dim += self.d_cxr
            if "ECG" in self.modeltype:
                dim += self.d_ecg      

            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)
        else:
            # baseline fusion methods
            self.d_txt = args.embed_dim
            self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
            self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

            if self.cross_method=="MulT":
                self.trans_txt_with_ts=self.get_network(self_type='txt_with_ts',layers=args.cross_layers)
                self.trans_ts_with_txt=self.get_network(self_type='ts_with_txt',layers=args.cross_layers)
                self.proj1 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                self.proj2 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
                self.out_layer = nn.Linear((self.d_ts+self.d_txt), output_dim)
            elif self.cross_method=="MAGGate":
                self.gate_fusion=MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.dropout)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            elif self.cross_method=="Outer":
                self.outer_fusion=Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)
                self.proj1 = nn.Linear(self.d_txt, self.d_txt)
                self.proj2 = nn.Linear(self.d_txt, self.d_txt)
                self.out_layer = nn.Linear(self.d_txt, output_dim)
            else:
                self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
                self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)
        
        # TODO: add baseline fusion methods for TS_CXR
        # if self.modeltype == "TS_CXR":
        #     if self.cross_method in ["self_cross", "moe", "moe_cross"]:
        #         self.trans_self_cross_ts_txt=self.get_cross_network(args, layers=args.cross_layers)
        #         self.proj1 = nn.Linear(self.d_ts+self.d_cxr, self.d_ts+self.d_cxr)
        #         self.proj2 = nn.Linear(self.d_ts+self.d_cxr, self.d_ts+self.d_cxr)
        #         self.out_layer = nn.Linear(self.d_ts+self.d_cxr, output_dim)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1=nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ts_mem', layers=-1):
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.ts_seq_num, None
        elif self_type == 'txt_mem':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, None
            else:
                embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.text_seq_num, None

        elif self_type =='txt_with_ts':
            if self.irregular_learn_emb_ts:
                embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.text_seq_num, self.ts_seq_num

        elif self_type =='ts_with_txt':
            if self.irregular_learn_emb_text:
                embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.tt_max, self.tt_max
            else:
                embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.ts_seq_num, self.text_seq_num
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=layers,
                                  device=self.device,
                                  attn_dropout=self.dropout,
                                  relu_dropout=self.dropout,
                                  res_dropout=self.dropout,
                                  embed_dropout=self.dropout,
                                  attn_mask=self.attn_mask,
                                  q_seq_len=q_seq_len,
                                  kv_seq_len=kv_seq_len)

    def get_cross_network(self, args, layers=-1):
        embed_dim, q_seq_len = self.d_ts, self.tt_max
        # not specified kv_seq_len, because this depends on the number of modalities
        return TransformerCrossEncoder(args=args,
                                        embed_dim=embed_dim,
                                        num_heads=self.num_heads,
                                        layers=layers,
                                        device=self.device,
                                        attn_dropout=self.dropout,
                                        relu_dropout=self.dropout,
                                        res_dropout=self.dropout,
                                        embed_dropout=self.dropout,
                                        attn_mask=self.attn_mask,
                                        q_seq_len_1=q_seq_len,
                                        num_modalities=self.num_modalities)

    def learn_time_embedding(self, tt):
        '''
        Time2Vec Module
        '''
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        # only two dimension?
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    def forward(self, x_ts, x_ts_mask, ts_tt_list, cxr_missing=None, text_missing=None, ecg_missing=None, note_missing = None, input_ids_sequences=None,
                attn_mask_sequences=None, text_emb=None, note_time_list=None, note_time_mask_list=None, note_embed=None,
                labels=None, reg_ts=None, cxr_feats=None, cxr_time=None, cxr_time_mask=None, ecg_feats=None,
                ecg_time=None, ecg_time_mask=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        if "TS" in self.modeltype:
            # mTAND module part
            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                #TODO: what does ts_mask used for?
                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)
                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts != None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                # print('proj_x_ts_reg', torch.isnan(proj_x_ts_reg).any())
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                # TODO: should we cluster reg or irg time series or both, check original paper?
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg, dim=0).values
                    g_reg =torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate=torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")
                # print('moe_gate', torch.isnan(moe_gate).any())
                mixup_rate = self.moe(moe_gate)
                # print('mixup_rate', torch.isnan(mixup_rate).any())
                proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")
            proj_x_ts += self.token_type_embeddings(torch.zeros((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))

        mod_count = 1
        if "Text" in self.modeltype:
            # compute irregular clinical notes attention
            # if text_missing is None or torch.all(text_missing == 0):
            if self.use_pt_text_embeddings:
                x_txt = text_emb
            else:
                x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)

            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_txt=self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt=proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)
            if text_missing is None or torch.all(text_missing == 0):
                proj_x_txt += self.token_type_embeddings(torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(text_missing == 0):
                missing_indices, non_missing = self._missing_indices(text_missing)
                proj_x_txt[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_txt[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if "CXR" in self.modeltype:
            # compute irregular clinical notes attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(cxr_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_cxr=self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
                proj_x_cxr=proj_x_cxr.transpose(0, 1)
            else:
                cxr_feats = cxr_feats.transpose(1, 2)
                proj_x_cxr = cxr_feats if self.orig_d_cxr == self.d_cxr else self.proj_cxr(cxr_feats)
                proj_x_cxr = proj_x_cxr.permute(2, 0, 1)
            if cxr_missing is None or torch.all(cxr_missing == 0):
                proj_x_cxr += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=self.device))
                # res = self.token_type_embeddings(torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=self.device))
                # proj_x_cxr += res
            elif not torch.all(cxr_missing == 0):
                # proj_x_cxr = None
                missing_indices, non_missing = self._missing_indices(cxr_missing)
                proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_cxr[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if "ECG" in self.modeltype:
            # compute irregular ECG attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(ecg_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_ecg=self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
                proj_x_ecg=proj_x_ecg.transpose(0, 1)
            else:
                ecg_feats = ecg_feats.transpose(1, 2)
                proj_x_ecg = ecg_feats if self.orig_d_ecg == self.d_ecg else self.proj_ecg(ecg_feats)
                proj_x_ecg = proj_x_ecg.permute(2, 0, 1)
            
            if ecg_missing is None or torch.all(ecg_missing == 0):
                proj_x_ecg += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(ecg_missing == 0):
                # proj_x_ecg = None
                missing_indices, non_missing = self._missing_indices(ecg_missing)
                proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_ecg[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if self.cross_method in ["self_cross", "moe", "hme"]:
            if self.modeltype == "TS_Text":
                hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_txt], ['ts', 'txt'])
            elif self.modeltype == "TS_CXR":
                hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr], ['ts', 'cxr'])
            elif self.modeltype == "TS_ECG":
                hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_ecg], ['ts', 'ecg'])
            elif self.modeltype == "TS_CXR_Text":
                hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt], ['ts', 'cxr', 'txt'])
            elif self.modeltype == "TS_CXR_ECG":
                hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_ecg], ['ts', 'cxr', 'ecg'])
            elif self.modeltype == "TS_ECG_Text":
                hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_txt, proj_x_ecg], ['ts', 'txt', 'ecg'])
            # elif self.modeltype == "TS_CXR_Text_Notes":
            #     hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt, proj_x_note], ['ts', 'cxr', 'txt', 'note'])
            elif self.modeltype == "TS_CXR_Text_ECG":
                hiddens = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt, proj_x_ecg], ['ts', 'cxr', 'txt', 'ecg'])
                
            if hiddens is None:
                return None
            # h_txt_with_ts, h_ts_with_txt=hiddens
            last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)
            # last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)
        else:
            if 'CXR' in self.modeltype:
                proj_x_txt = proj_x_cxr
            if self.cross_method=="MulT":
                # ts --> txt
                h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
                # txt --> ts
                h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
                proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
                proj_x_txt = self.trans_txt_mem(h_ts_with_txt)

                last_h_ts=proj_x_ts[-1]
                last_h_txt=proj_x_txt[-1]
                last_hs = torch.cat([last_h_ts,last_h_txt], dim=1)

            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
                proj_x_txt = self.trans_txt_mem(proj_x_txt)
                if self.cross_method=="MAGGate":
                    last_hs=self.gate_fusion(proj_x_txt[-1],proj_x_ts[-1])
                elif self.cross_method=="Outer":
                    last_hs=self.outer_fusion(proj_x_txt[-1],proj_x_ts[-1])
                else:
                    last_hs = torch.cat([proj_x_txt[-1],proj_x_ts[-1]], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        # if 'ihm' in self.task or 'los' in self.task:
        #     if labels!=None:
        #         return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
        #     return last_hs_proj, torch.nn.functional.softmax(output,dim=-1)[:,1]

        if 'ihm' in self.task:
            if labels is not None:
                # Continue using binary classification approach for IHM
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)[:, 1]

        elif 'los' in self.task:
            if labels is not None:
                # Use a multiclass classification approach for LOS
                # Assuming output has the correct shape (batch_size, num_classes)
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return last_hs_proj, torch.nn.functional.sigmoid(output)


class TSMixed(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ts=None,orig_reg_d_ts=None,ts_seq_num=None):

        super(TSMixed, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.num_heads = args.num_heads

        self.attn_mask = False
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        self.irregular_learn_emb_ts=args.irregular_learn_emb_ts
        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.Interp=args.Interp
        self.reg_ts=args.reg_ts
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level
        self.task=args.task
        self.TS_model=args.TS_model
        self.tt_max=args.tt_max

        if self.irregular_learn_emb_text or self.irregular_learn_emb_ts:
            self.time_query=torch.linspace(0, 1., self.tt_max)
            self.periodic = nn.Linear(1, args.embed_time-1)
            self.linear = nn.Linear(1, 1)

        output_dim = args.num_labels

        self.orig_d_ts=orig_d_ts
        self.d_ts=args.embed_dim
        self.ts_seq_num=ts_seq_num

        if self.Interp:
            self.s_intp=S_Interp(args,self.device,self.orig_d_ts)
            self.c_intp=Cross_Interp(args,self.device,self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.irregular_learn_emb_ts:
            self.time_attn_ts=multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)

        if self.reg_ts:
            self.orig_reg_d_ts=orig_reg_d_ts
            self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.TS_mixup:
            if self.mixup_level=='batch':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=self.dropout)
            elif self.mixup_level=='batch_seq_feature':
                self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=self.dropout)
            else:
                raise ValueError("Unknown mixedup type")

                # self.moe = nn.Linear(self.d_ts*self.tt_max*2, 1)
        if self.TS_model=='LSTM':
            self.trans_ts_mem=nn.LSTM(input_size=self.d_ts, hidden_size=self.d_ts, num_layers=args.layers,dropout=self.dropout,bidirectional=True)

        elif self.TS_model=='CNN':
            self.trans_ts_mem=TimeSeriesCnnModel(input_size=self.d_ts,n_filters=self.d_ts,filter_size=self.kernel_size,\
            dropout=self.dropout,length=self.tt_max,n_neurons=self.d_ts,layers=args.layers)
        elif self.TS_model=='Atten':
            self.trans_ts_mem = self.get_network(args=args, self_type='ts_mem', layers=args.layers)
        
        self.proj1 = nn.Linear(self.d_ts, self.d_ts)
        self.proj2 = nn.Linear(self.d_ts, self.d_ts)
        self.out_layer= nn.Linear(self.d_ts, output_dim)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1=nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, args, self_type='ts_mem', layers=-1):
        embed_dim=self.d_ts
        if self_type == 'ts_mem':
            if self.irregular_learn_emb_ts :
                q_seq_len= self.tt_max
            else:
                q_seq_len= self.ts_seq_num

        return TransformerEncoderMoE(args=args,
                                    embed_dim=embed_dim,
                                    num_heads=self.num_heads,
                                    layers=layers,
                                    device=self.device,
                                    attn_dropout=self.dropout,
                                    relu_dropout=self.dropout,
                                    res_dropout=self.dropout,
                                    embed_dropout=self.dropout,
                                    attn_mask=self.attn_mask,
                                    q_seq_len=q_seq_len,
                                    kv_seq_len=None)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x_ts, x_ts_mask, ts_tt_list,labels=None,reg_ts=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        if "TS" in self.modeltype :

            if self.Interp:
                x_ts_mask_interp=copy.deepcopy(x_ts_mask)
                x_ts_interp=copy.deepcopy(x_ts)
                recon_m=hold_out(x_ts_mask_interp)
                recon_m=torch.Tensor(recon_m).to(self.device)
                proj_x_ts_interp=self.proj_ts_intp(self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m))) #dimension [batch_size,  n_features,seq_len]
                proj_x_ts_interp = proj_x_ts_interp.permute(2, 0, 1)
                recon_interp=self.c_intp(self.s_intp(x_ts_interp, x_ts_mask_interp, ts_tt_list,recon_m, reconstruction=True),reconstruction=True)

            if self.irregular_learn_emb_ts:
                time_key_ts = self.learn_time_embedding(ts_tt_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                x_ts_irg = torch.cat((x_ts,x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask,x_ts_mask), 2)

                proj_x_ts_irg=self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg=proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts!=None:
                x_ts_reg = reg_ts.transpose(1, 2)
                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts== self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.Interp and not self.irregular_learn_emb_ts and self.reg_ts:
                    proj_x_ts_irg=proj_x_ts_interp
                if self.Interp and self.irregular_learn_emb_ts and not self.reg_ts :
                    proj_x_ts_reg=proj_x_ts_interp
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg,dim=0).values
                    g_reg =torch.max(proj_x_ts_reg,dim=0).values
                    moe_gate=torch.cat([g_irg,g_reg],dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")

                # for name, parameter in self.moe.named_parameters():
                mixup_rate=self.moe(moe_gate)
                proj_x_ts=mixup_rate*proj_x_ts_irg+(1-mixup_rate)*proj_x_ts_reg

            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts=proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts=proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")


            if self.TS_model=='CNN':
                proj_x_ts = proj_x_ts.permute(1, 2, 0)
                proj_x_ts = self.trans_ts_mem(proj_x_ts)

            elif self.TS_model=='LSTM':
                    _, (proj_x_ts, _) = self.trans_ts_mem(proj_x_ts)
            else:
                proj_x_ts = self.trans_ts_mem(proj_x_ts)
            if  self.TS_model!='CNN':
                last_h_ts=proj_x_ts[-1]

            else:
                last_h_ts=proj_x_ts

 
            if self.modeltype=="TS" :
                last_hs=last_h_ts
            else:
                raise ValueError("Unknown model type")
                       
            last_hs = torch.mean(proj_x_ts[-1], dim=0)
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        if self.Interp:
            reconloss_interp=recon_loss(x_ts_interp,x_ts_mask_interp,recon_m,recon_interp,self.d_ts)

        # if 'ihm' in self.task or 'los' in self.task:
        #     if labels!=None:
        #         if self.Interp:
        #             return self.loss_fct1(output, labels)+reconloss_interp
        #         else:
        #             return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
        #     return last_hs_proj, torch.nn.functional.softmax(output,dim=-1)[:,1]

        if 'ihm' in self.task:
            if labels is not None:
                if self.Interp:
                    # Assuming `reconloss_interp` is calculated elsewhere in the code
                    return self.loss_fct1(output, labels) + reconloss_interp
                else:
                    # For binary classification of IHM, assuming output is two classes
                    return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
            else:
                # If labels are None, return the projection and the softmax probability of the positive class
                return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)[:, 1]

        elif 'los' in self.task:
            if labels is not None:
                if self.Interp:
                    # Handle interpolation logic if applicable to LOS task
                    return self.loss_fct1(output, labels) + reconloss_interp
                else:
                    # For multiclass classification, return the loss and the softmax probabilities for all classes
                    return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)
            else:
                # If labels are None, return the projection and the softmax probabilities for all classes
                return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                if self.Interp:
                    return self.loss_fct1(output, labels)+reconloss_interp
                else:
                    return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class TextMoE(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_txt=None, text_seq_num=None, Biobert=None):

        super(TextMoE, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.args = args
        self.num_modalities = args.num_modalities
        self.use_pt_text_embeddings = args.use_pt_text_embeddings
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level

        self.num_heads = args.num_heads
        self.attn_mask = False
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        
        self.Interp=args.Interp

        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr=args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg=args.irregular_learn_emb_ecg

        self.task=args.task

        self.tt_max=args.tt_max

        self.time_query=torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)

        output_dim = args.num_labels

        self.orig_d_txt=orig_d_txt
        self.d_txt=args.embed_dim
        self.text_seq_num=text_seq_num
        self.bertrep = BertForRepresentation(args, Biobert)

        if self.Interp:
            self.s_intp=S_Interp(args,self.device,self.orig_d_ts)
            self.c_intp=Cross_Interp(args,self.device,self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.modeltype == "Text": 
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn_text = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)    

            dim = self.d_txt
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)

        self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1=nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='txt_mem', layers=-1):
        embed_dim=self.d_txt
        if self.irregular_learn_emb_text:
            embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, None
        else:
            embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.text_seq_num, None

        return TransformerEncoderMoE(
                                args=self.args,
                                embed_dim=embed_dim,
                                num_heads=self.num_heads,
                                layers=layers,
                                device=self.device,
                                attn_dropout=self.dropout,
                                relu_dropout=self.dropout,
                                res_dropout=self.dropout,
                                embed_dropout=self.dropout,
                                attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                kv_seq_len=kv_seq_len)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    def forward(self, input_ids_sequences, attn_mask_sequences, 
                text_emb, note_time_list, note_time_mask_list, text_missing, labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        if "Text" in self.modeltype :
            if self.use_pt_text_embeddings:
                x_txt = text_emb
            else:
                x_txt = self.bertrep(input_ids_sequences, attn_mask_sequences)

            if self.irregular_learn_emb_text:
                time_key = self.learn_time_embedding(note_time_list).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                proj_x_txt=self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt=proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)

            if text_missing is None or torch.all(text_missing == 0):
                # Initialize `temp` with zeros to match the purpose of imputation
                temp = torch.zeros((self.args.tt_max, x_txt.shape[0]), dtype=torch.float, device=x_txt.device)

                # Add a dimension to `temp` to match `proj_x_txt`
                temp = temp.unsqueeze(-1)  # Shape is now `[tt_max, batch_size, 1]`

                # Ensure proj_x_txt and temp are on the same device before adding
                proj_x_txt = proj_x_txt.to(temp.device)
                proj_x_txt += temp
            elif not torch.all(text_missing == 0):
                missing_indices, non_missing = self._missing_indices(text_missing)
                proj_x_txt[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_txt.device))
                proj_x_txt[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_txt.device)

            proj_x_txt = self.trans_txt_mem(proj_x_txt)
            # For a single modality, no concatenation is required, directly use the output
            #last_hs = proj_x_txt[-1]  # Use the last hidden state of the single modality
            last_hs = torch.mean(proj_x_txt[-1], dim=0)  # Average over time dimension [batch_size, embed_dim]

            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        # if 'ihm' in self.task or 'los' in self.task:
        #     if labels!=None:
        #         return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
        #     return last_hs_proj, torch.nn.functional.softmax(output,dim=-1)[:,1]
        if 'ihm' in self.task:
            if labels is not None:
                # Continue using binary classification approach for IHM
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)[:, 1]

        elif 'los' in self.task:
            if labels is not None:
                # Use a multiclass classification approach for LOS
                # Assuming output has the correct shape (batch_size, num_classes)
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class CXRMoE(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_cxr=None):

        super(CXRMoE, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.args = args
        self.num_modalities = args.num_modalities
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level

        self.num_heads = args.num_heads
        self.attn_mask = False
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        
        self.Interp=args.Interp

        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr=args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg=args.irregular_learn_emb_ecg

        self.task=args.task

        self.tt_max=args.tt_max

        self.time_query=torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)

        output_dim = args.num_labels

        self.orig_d_cxr=orig_d_cxr
        self.d_cxr=args.embed_dim

        if self.Interp:
            self.s_intp=S_Interp(args,self.device,self.orig_d_ts)
            self.c_intp=Cross_Interp(args,self.device,self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.modeltype == "CXR":
            self.orig_d_cxr = 1024
            self.d_cxr = args.embed_dim
            self.cxr_seq_num = 5

            if self.irregular_learn_emb_cxr:
                self.time_attn_cxr = multiTimeAttention(1024, self.d_cxr, args.embed_time, 8)
            else:
                self.proj_cxr = nn.Conv1d(self.orig_d_cxr, self.d_cxr, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
                
            dim = self.d_cxr
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)

        self.trans_cxr_mem = self.get_network(self_type='cxr_mem', layers=args.layers)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1=nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='cxr_mem', layers=-1):
        embed_dim=self.d_cxr
        if self.irregular_learn_emb_cxr:
            embed_dim, q_seq_len, kv_seq_len = self.d_cxr, self.tt_max, None
        else:
            embed_dim, q_seq_len, kv_seq_len = self.d_cxr, self.cxr_seq_num, None

        return TransformerEncoderMoE(
                                args=self.args,
                                embed_dim=embed_dim,
                                num_heads=self.num_heads,
                                layers=layers,
                                device=self.device,
                                attn_dropout=self.dropout,
                                relu_dropout=self.dropout,
                                res_dropout=self.dropout,
                                embed_dropout=self.dropout,
                                attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                kv_seq_len=kv_seq_len)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    def forward(self, cxr_feats, cxr_time, cxr_time_mask, cxr_missing, labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        if "CXR" in self.modeltype:
            # compute irregular clinical notes attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(cxr_time).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                proj_x_cxr=self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
                proj_x_cxr=proj_x_cxr.transpose(0, 1)
            else:
                cxr_feats = cxr_feats.transpose(1, 2)
                proj_x_cxr = cxr_feats if self.orig_d_cxr == self.d_cxr else self.proj_cxr(cxr_feats)
                proj_x_cxr = proj_x_cxr.permute(2, 0, 1)
            if cxr_missing is None or torch.all(cxr_missing == 0):
                # Initialize `temp` with zeros to match the purpose of imputation
                temp = torch.zeros((self.args.tt_max, cxr_feats.shape[0]), dtype=torch.float, device=cxr_feats.device)

                # Add a dimension to `temp` to match `proj_x_txt`
                temp = temp.unsqueeze(-1)  # Shape is now `[tt_max, batch_size, 1]`

                # Ensure proj_x_txt and temp are on the same device before adding
                proj_x_cxr = proj_x_cxr.to(temp.device)
                proj_x_cxr += temp
            elif not torch.all(cxr_missing == 0):
                # proj_x_cxr = None
                missing_indices, non_missing = self._missing_indices(cxr_missing)
                proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(1 * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_cxr[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            
            proj_x_cxr = self.trans_cxr_mem(proj_x_cxr)
            # For a single modality, no concatenation is required, directly use the output
            #last_hs = proj_x_txt[-1]  # Use the last hidden state of the single modality
            last_hs = torch.mean(proj_x_cxr[-1], dim=0)  # Average over time dimension [batch_size, embed_dim]

            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        # if 'ihm' in self.task or 'los' in self.task:
        #     if labels!=None:
        #         return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
        #     return last_hs_proj, torch.nn.functional.softmax(output,dim=-1)[:,1]
        if 'ihm' in self.task:
            if labels is not None:
                # Continue using binary classification approach for IHM
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)[:, 1]

        elif 'los' in self.task:
            if labels is not None:
                # Use a multiclass classification approach for LOS
                # Assuming output has the correct shape (batch_size, num_classes)
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)

class ECGMoE(nn.Module):
    def __init__(self,args,device,modeltype=None,orig_d_ecg=None):

        super(ECGMoE, self).__init__()
        if modeltype!=None:
            self.modeltype=modeltype
        else:
            self.modeltype=args.modeltype
        self.args = args
        self.num_modalities = args.num_modalities
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)
        self.TS_mixup=args.TS_mixup
        self.mixup_level=args.mixup_level

        self.num_heads = args.num_heads
        self.attn_mask = False
        self.layers = args.layers
        self.device=device
        self.kernel_size=args.kernel_size
        self.dropout=args.dropout
        
        self.Interp=args.Interp

        self.irregular_learn_emb_text=args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr=args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg=args.irregular_learn_emb_ecg

        self.task=args.task

        self.tt_max=args.tt_max

        self.time_query=torch.linspace(0, 1., self.tt_max)
        self.periodic = nn.Linear(1, args.embed_time-1)
        self.linear = nn.Linear(1, 1)

        output_dim = args.num_labels

        self.orig_d_ecg=orig_d_ecg
        self.d_ecg=args.embed_dim

        if self.Interp:
            self.s_intp=S_Interp(args,self.device,self.orig_d_ts)
            self.c_intp=Cross_Interp(args,self.device,self.orig_d_ts)
            self.proj_ts_intp = nn.Conv1d(self.orig_d_ts*3, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if "ECG" in self.modeltype:
            self.orig_d_ecg = 256
            self.d_ecg = args.embed_dim
            self.ecg_seq_num = 5

            if self.irregular_learn_emb_ecg:
                self.time_attn_ecg = multiTimeAttention(256, self.d_ecg, args.embed_time, 8)
            else:
                self.proj_ecg = nn.Conv1d(self.orig_d_ecg, self.d_ecg, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
                
            dim = self.d_ecg
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)

        self.trans_ecg_mem = self.get_network(self_type='ecg_mem', layers=args.layers)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1=nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_network(self, self_type='ecg_mem', layers=-1):
        embed_dim=self.d_ecg
        if self.irregular_learn_emb_ecg:
            embed_dim, q_seq_len, kv_seq_len = self.d_ecg, self.tt_max, None
        else:
            embed_dim, q_seq_len, kv_seq_len = self.d_ecg, self.ecg_seq_num, None

        return TransformerEncoderMoE(
                                args=self.args,
                                embed_dim=embed_dim,
                                num_heads=self.num_heads,
                                layers=layers,
                                device=self.device,
                                attn_dropout=self.dropout,
                                relu_dropout=self.dropout,
                                res_dropout=self.dropout,
                                embed_dropout=self.dropout,
                                attn_mask=self.attn_mask,
                                q_seq_len=q_seq_len,
                                kv_seq_len=kv_seq_len)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
    
    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    def forward(self, ecg_feats, ecg_time, ecg_time_mask, ecg_missing, labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """
        if "ECG" in self.modeltype:
            # compute irregular ECG attention
            if self.irregular_learn_emb_ecg:
                time_key = self.learn_time_embedding(ecg_time).to(self.device)
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_ecg=self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
                proj_x_ecg=proj_x_ecg.transpose(0, 1)
            else:
                ecg_feats = ecg_feats.transpose(1, 2)
                proj_x_ecg = ecg_feats if self.orig_d_ecg == self.d_ecg else self.proj_ecg(ecg_feats)
                proj_x_ecg = proj_x_ecg.permute(2, 0, 1)
            
            if ecg_missing is None or torch.all(ecg_missing == 0):
            # Initialize `temp` with zeros to match the purpose of imputation
                temp = torch.zeros((self.args.tt_max, ecg_feats.shape[0]), dtype=torch.float, device=ecg_feats.device)

                # Add a dimension to `temp` to match `proj_x_txt`
                temp = temp.unsqueeze(-1)  # Shape is now `[tt_max, batch_size, 1]`

                # Ensure proj_x_txt and temp are on the same device before adding
                proj_x_ecg = proj_x_ecg.to(temp.device)
                proj_x_ecg += temp
            elif not torch.all(ecg_missing == 0):
                # proj_x_ecg = None
                missing_indices, non_missing = self._missing_indices(ecg_missing)
                proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_ecg[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)

            proj_x_ecg = self.trans_ecg_mem(proj_x_ecg)
            # For a single modality, no concatenation is required, directly use the output
            #last_hs = proj_x_txt[-1]  # Use the last hidden state of the single modality
            last_hs = torch.mean(proj_x_ecg[-1], dim=0)  # Average over time dimension [batch_size, embed_dim]

            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        # if 'ihm' in self.task or 'los' in self.task:
        #     if labels!=None:
        #         return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
        #     return last_hs_proj, torch.nn.functional.softmax(output,dim=-1)[:,1]
        if 'ihm' in self.task:
            if labels is not None:
                # Continue using binary classification approach for IHM
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)[:, 1]

        elif 'los' in self.task:
            if labels is not None:
                # Use a multiclass classification approach for LOS
                # Assuming output has the correct shape (batch_size, num_classes)
                return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)
            return last_hs_proj, torch.nn.functional.softmax(output, dim=-1)

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)