import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from core.module import *
from core.interp import *
import copy
import pdb


class MULTCrossModel(nn.Module):
    def __init__(self,args,device,modeltype=None, orig_d_txt=None,orig_d_visual=None,orig_d_acoustic=None,seq_num=None, Biobert=None):
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
        self.task=args.task
        self.tt_max=args.tt_max
        self.cross_method=args.cross_method
        self.num_modalities = args.num_modalities

        if 'Text' in self.modeltype:
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim
            self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if 'Visual' in self.modeltype:
            self.orig_d_visual = orig_d_visual
            self.d_visual = args.embed_dim
            self.proj_visual = nn.Conv1d(self.orig_d_visual, self.d_visual, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if 'Acoustic' in self.modeltype:
            self.orig_d_acoustic = orig_d_acoustic
            self.d_acoustic = args.embed_dim
            self.proj_acoustic = nn.Conv1d(self.orig_d_acoustic, self.d_acoustic, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = args.num_labels

        if self.cross_method in ["self_cross", "moe", "hme"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.layers)
            dim = args.embed_dim * self.num_modalities
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)
        # else:
        #     # baseline fusion methods
        #     self.d_txt = args.embed_dim
        #     self.trans_ts_mem = self.get_network(self_type='ts_mem', layers=args.layers)
        #     self.trans_txt_mem = self.get_network(self_type='txt_mem', layers=args.layers)

        #     if self.cross_method=="MulT":
        #         self.trans_txt_with_ts=self.get_network(self_type='txt_with_ts',layers=args.cross_layers)
        #         self.trans_ts_with_txt=self.get_network(self_type='ts_with_txt',layers=args.cross_layers)
        #         self.proj1 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
        #         self.proj2 = nn.Linear((self.d_ts+self.d_txt), (self.d_ts+self.d_txt))
        #         self.out_layer = nn.Linear((self.d_ts+self.d_txt), output_dim)
        #     elif self.cross_method=="MAGGate":
        #         self.gate_fusion=MAGGate(inp1_size=self.d_txt, inp2_size=self.d_ts, dropout=self.dropout)
        #         self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        #         self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        #         self.out_layer = nn.Linear(self.d_txt, output_dim)
        #     elif self.cross_method=="Outer":
        #         self.outer_fusion=Outer(inp1_size=self.d_txt, inp2_size=self.d_ts)
        #         self.proj1 = nn.Linear(self.d_txt, self.d_txt)
        #         self.proj2 = nn.Linear(self.d_txt, self.d_txt)
        #         self.out_layer = nn.Linear(self.d_txt, output_dim)
        #     else:
        #         self.proj1 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        #         self.proj2 = nn.Linear(self.d_ts+self.d_txt, self.d_ts+self.d_txt)
        #         self.out_layer = nn.Linear(self.d_ts+self.d_txt, output_dim)

        if 'multiclass' in self.task:
            #self.loss_fct1=nn.CrossEntropyLoss()
            #self.loss_fct1 = nn.MSELoss()
            self.loss_fct1 = nn.L1Loss(reduction="mean")
        elif 'multilabel' in self.task:
            self.loss_fct1=nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    # def get_network(self, self_type='ts_mem', layers=-1):
    #     if self_type == 'ts_mem':
    #         if self.irregular_learn_emb_ts:
    #             embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.tt_max, None
    #         else:
    #             embed_dim, q_seq_len, kv_seq_len = self.d_ts, self.ts_seq_num, None
    #     elif self_type == 'txt_mem':
    #         if self.irregular_learn_emb_text:
    #             embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.tt_max, None
    #         else:
    #             embed_dim, q_seq_len, kv_seq_len = self.d_txt, self.text_seq_num, None

    #     elif self_type =='txt_with_ts':
    #         if self.irregular_learn_emb_ts:
    #             embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.tt_max, self.tt_max
    #         else:
    #             embed_dim, q_seq_len,kv_seq_len = self.d_ts, self.text_seq_num, self.ts_seq_num

    #     elif self_type =='ts_with_txt':
    #         if self.irregular_learn_emb_text:
    #             embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.tt_max, self.tt_max
    #         else:
    #             embed_dim, q_seq_len,kv_seq_len = self.d_txt, self.ts_seq_num, self.text_seq_num
    #     else:
    #         raise ValueError("Unknown network type")

    #     return TransformerEncoder(embed_dim=embed_dim,
    #                               num_heads=self.num_heads,
    #                               layers=layers,
    #                               device=self.device,
    #                               attn_dropout=self.dropout,
    #                               relu_dropout=self.dropout,
    #                               res_dropout=self.dropout,
    #                               embed_dropout=self.dropout,
    #                               attn_mask=self.attn_mask,
    #                               q_seq_len=q_seq_len,
    #                               kv_seq_len=kv_seq_len)

    def get_cross_network(self, args, layers=-1):
        embed_dim, q_seq_len = args.embed_dim, args.tt_max
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


    def forward(self, text_features, visual_features, acoustic_features, labels=None):
        """
        dimension [batch_size, seq_len, n_features]

        """

        if "Text" in self.modeltype:
            # compute irregular clinical notes attention
            # if text_missing is None or torch.all(text_missing == 0):
            x_txt = text_features.transpose(1,2)
            proj_x_txt = self.proj_txt(x_txt)
            proj_x_txt = F.interpolate(proj_x_txt, size=self.tt_max, mode='linear', align_corners=False)
            proj_x_txt = proj_x_txt.permute(2, 0, 1)

        if "Visual" in self.modeltype:
            x_visual = visual_features.transpose(1,2)
            proj_x_visual = self.proj_visual(x_visual)
            proj_x_visual = F.interpolate(proj_x_visual, size=self.tt_max, mode='linear', align_corners=False)
            proj_x_visual = proj_x_visual.permute(2, 0, 1)

        if "Acoustic" in self.modeltype:
            x_acoustic = acoustic_features.transpose(1,2)
            proj_x_acoustic = self.proj_acoustic(x_acoustic)
            proj_x_acoustic = F.interpolate(proj_x_acoustic, size=self.tt_max, mode='linear', align_corners=False)
            proj_x_acoustic = proj_x_acoustic.permute(2, 0, 1)

        if self.cross_method in ["self_cross", "moe", "hme"]:
            if self.modeltype == "Text_Visual_Acoustic":
                hiddens = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_visual, proj_x_acoustic], ['txt', 'visual', 'acoustic'])

            if hiddens is None:
                return None
            # h_txt_with_ts, h_ts_with_txt=hiddens
            last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)
            # last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)
        # else:
        #     if 'CXR' in self.modeltype:
        #         proj_x_txt = proj_x_cxr
        #     if self.cross_method=="MulT":
        #         # ts --> txt
        #         h_txt_with_ts = self.trans_txt_with_ts(proj_x_txt, proj_x_ts, proj_x_ts)
        #         # txt --> ts
        #         h_ts_with_txt = self.trans_ts_with_txt(proj_x_ts, proj_x_txt, proj_x_txt)
        #         proj_x_ts = self.trans_ts_mem(h_txt_with_ts)
        #         proj_x_txt = self.trans_txt_mem(h_ts_with_txt)

        #         last_h_ts=proj_x_ts[-1]
        #         last_h_txt=proj_x_txt[-1]
        #         last_hs = torch.cat([last_h_ts,last_h_txt], dim=1)

        #     else:
        #         proj_x_ts = self.trans_ts_mem(proj_x_ts)
        #         proj_x_txt = self.trans_txt_mem(proj_x_txt)
        #         if self.cross_method=="MAGGate":
        #             last_hs=self.gate_fusion(proj_x_txt[-1],proj_x_ts[-1])
        #         elif self.cross_method=="Outer":
        #             last_hs=self.outer_fusion(proj_x_txt[-1],proj_x_ts[-1])
        #         else:
        #             last_hs = torch.cat([proj_x_txt[-1],proj_x_ts[-1]], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        # if 'ihm' in self.task or 'los' in self.task:
        #     if labels!=None:
        #         return self.loss_fct1(output, labels), torch.nn.functional.softmax(output, dim=-1)[:, 1]
        #     return last_hs_proj, torch.nn.functional.softmax(output,dim=-1)[:,1]

        if 'multiclass' in self.task:
            if labels is not None:
                # Directly use output for regression; no softmax or class conversion needed
                return self.loss_fct1(output, labels), output  # Return loss and raw output predictions
            return output

        # elif 'multilabel' in self.task:
        #     if labels!=None:
        #         return self.loss_fct1(output, labels)
        #     return last_hs_proj, torch.nn.functional.sigmoid(output)


