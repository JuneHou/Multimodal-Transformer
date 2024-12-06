class TransformerEncoderLayerMoE(nn.Module):
    def __init__(self, args, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, 
                 embed_dropout=0.0, attn_mask=False, num_modalities=1):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.attn_mask = attn_mask

        # Layer Normalizations for self-attention mechanisms
        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])
        self.post_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])
        self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        self.post_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        # Self-attention mechanism
        self.self_attns = nn.ModuleList([
            MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_dropout) 
            for _ in range(num_modalities)
        ])

        self.attn_mask = attn_mask
        # Dropout settings
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

        # Feedforward layers
        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])
        self.fc1 = nn.ModuleList([nn.Linear(embed_dim, 4 * embed_dim) for _ in range(num_modalities)])
        self.fc2 = nn.ModuleList([nn.Linear(4 * embed_dim, embed_dim) for _ in range(num_modalities)])
        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])


   if args.cross_method == 'moe':
            moe_config = MoEConfig(
            num_experts=args.num_of_experts[0],
            moe_input_size=args.tt_max * args.embed_dim * num_modalities,
            moe_hidden_size=args.hidden_size,
            moe_output_size=args.tt_max * args.embed_dim * num_modalities,
            top_k=args.top_k[0],
            router_type=args.router_type,
            num_modalities=args.num_modalities,
            gating=args.gating_function[0])
            self.moe = MoE(moe_config)
            self.moe = self.moe.to(args.device)
    elif args.cross_method == 'hme':
        moe_config = MoEConfig(
        num_experts=args.num_of_experts,
        moe_input_size=args.tt_max * args.embed_dim * num_modalities,
        moe_hidden_size=args.hidden_size,
        moe_output_size=args.tt_max * args.embed_dim * num_modalities,
        top_k=args.top_k,
        router_type=args.router_type,
        num_modalities=args.num_modalities,
        gating=args.gating_function)
        self.moe = HierarchicalMoE(moe_config)
        self.moe = self.moe.to(args.device)

    def forward(self, x_list, modality):
        """
        Args:
            x_list (List of Tensor): List of inputs for each modality, shape `(seq_len, batch, embed_dim)`
        """
        residual = x_list
        seq_len, bs = x_list[0].shape[0], x_list[0].shape[1]

        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]

        #  a list of tuples, where each tuple contains (attn, attn_weights) from each respective MultiheadAttention module.
        output = [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]
        # attn: output[0][0].shape -> [48, 3, 128]; attn_weights: output[0][1].shape -> [3, 48, 48]
        # filter out attn_weights
        x_list = [x for x, _ in output]
        x_list = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list]
        x_list = [r + x for r, x in zip(residual, x_list)]

        # moe or cross attn
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        if self.args.cross_method in ["moe", "hme"]:
            x_mod_in = [torch.reshape(x, (bs, -1)) for x in x_list]
            embd_len_list = [0] + list(np.cumsum([x.shape[1] for x in x_mod_in]))
            embeddings = torch.concat(x_mod_in, dim=1)
            if torch.isnan(embeddings).any():
                return None
            # just replace this with hierarchical moe
            moe_out, balance_loss = self.moe(x_mod_in, modalities=modality)
            x_mod_out = [moe_out[:, embd_len_list[i]:embd_len_list[i + 1]] for i in range(len(embd_len_list) - 1)]
            x_allmod_output = [torch.reshape(x, (seq_len, bs, -1)) for x in x_mod_out]
            moe_output = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_allmod_output]
            x_list = [r + x for r, x in zip(residual, moe_output)]


        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]
        x_list = [F.dropout(x, p=self.relu_dropout, training=self.training) for x in x_list]
        x_list = [l(x) for l, x in zip(self.fc2, x_list)]
        x_list = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list]
        x_list = [r + x  for r, x in zip(residual, x_list) ]
        return x_list

class TransformerEncoderMoE(nn.Module):
    def __init__(self, args, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, q_seq_len=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.embed_scale = math.sqrt(embed_dim)
        self.dropout = embed_dropout
        
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayerMoE(args,
                                                    embed_dim,
                                                    num_heads=num_heads,
                                                    attn_dropout=attn_dropout,
                                                    relu_dropout=relu_dropout,
                                                    res_dropout=res_dropout,
                                                    attn_mask=attn_mask,
                                                    num_modalities=1)
            self.layers.append(new_layer)
        
        self.embed_positions = nn.Embedding(q_seq_len, embed_dim, padding_idx=0) if q_seq_len else None
        nn.init.normal_(self.embed_positions.weight, std=0.02) if q_seq_len else None
        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_list = x_in_list
        lengths, positions = [], []
        lengths.append(x_list[0].size(0))
        x_list = [self.embed_scale * x_in for x_in in x_in_list]
        for length in lengths:
            positions.append(torch.tensor(torch.arange(length),dtype=torch.long).to(self.device))
        x_list = [l(position_x).unsqueeze(0).transpose(0,1) + x for l, x, position_x in zip(self.embed_positions_q, x_list, positions)]
            # Add positional embedding
        x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]

        for layer in self.layers:
            x_list = layer(x_list, modality) #proj_x_txt, proj_x_ts
            # len(x_list) = 2
            # x_list[0].shape = torch.Size([48, 1, 128])
            if x_list is None:
                return None

        if self.normalize:
            x_list=[l(x) for l, x in zip(self.layer_norm, x_list)]
        return x_list

