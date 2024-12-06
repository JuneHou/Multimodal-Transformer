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

        if "Text" in self.modeltype: 
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim
            self.text_seq_num = text_seq_num
            self.bertrep = BertForRepresentation(args, Biobert)

            if self.irregular_learn_emb_text:
                self.time_attn_text = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if self.cross_method in ["self_cross", "moe", "hme"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.cross_layers)
            dim = 0
            if self.modeltype == "TS":
                dim = self.d_ts
            if self.modeltype == "Text":
                dim = self.d_txt
            if self.modeltype == "CXR":
                dim = self.d_cxr
            if self.modeltype == "ECG":
                dim = self.d_ecg      

            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)

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
            
            if self.cross_method in ["self_cross", "moe", "hme"]:
                if self.modeltype == "TS":
                    # Single modality (time series) case
                    hiddens = self.trans_self_cross_ts_txt([proj_x_ts], ['ts'])
                elif self.modeltype == "Text":
                    # Single modality (text) case
                    hiddens = self.trans_self_cross_ts_txt([proj_x_txt], ['txt'])
                elif self.modeltype == "CXR":
                    # Single modality (CXR images) case
                    hiddens = self.trans_self_cross_ts_txt([proj_x_cxr], ['cxr'])
                elif self.modeltype == "ECG":
                    # Single modality (ECG signals) case
                    hiddens = self.trans_self_cross_ts_txt([proj_x_ecg], ['ecg'])
                else:
                    raise ValueError(f"Unsupported single-modality model type: {self.modeltype}")

                if hiddens is None:
                    return None

                # For a single modality, no concatenation is required, directly use the output
                last_hs = hiddens[-1]  # Use the last hidden state of the single modality

            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)

        if 'ihm' in self.task or 'los' in self.task:
            if labels!=None:
                return self.loss_fct1(output, labels)
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                return self.loss_fct1(output, labels)
            return torch.nn.functional.sigmoid(output)