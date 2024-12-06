#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils.config import MoEConfig
from core.sparse_moe import MoE
from core.hme import HierarchicalMoE
import sys
import pdb


class Outer(nn.Module):
    def __init__(self,
                 inp1_size: int = 128,
                 inp2_size: int = 128,
                 n_neurons: int = 128):
        super(Outer, self).__init__()
        self.inp1_size = inp1_size
        self.inp2_size = inp2_size
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
        )

    def forward(self, inp1, inp2):
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1)).type_as(inp1)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)
        fusion = torch.zeros((batch_size, self.inp1_size + 1, self.inp2_size + 1)).type_as(inp1)
        for  i  in  range ( batch_size ):
            fusion[i] = torch.outer(inp1[i], inp2[i])
        fusion = fusion.flatten(1)

        return self.feedforward(fusion)



class MAGGate(nn.Module):
    def __init__(self, inp1_size, inp2_size, dropout):
        super(MAGGate, self).__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc3 = nn.Linear(inp2_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        adjust = self.fc3(w2 * inp2)
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output


class gateMLP(nn.Module):
    def __init__(self,input_dim,hidden_size,output_dim,dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
             nn.Dropout(dropout),
             nn.Linear(input_dim, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size,output_dim),
             nn.Sigmoid()
        )


        self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)


    def forward(self,hidden_states ):
        gate_logits = self.gate(hidden_states)
        return gate_logits


class TimeSeriesCnnModel(nn.Module):
    def __init__(self,input_size,n_filters,filter_size,dropout,length,n_neurons,layers):
        super().__init__()

        padding = int(np.floor(filter_size / 2))
        self.layers=layers
        if layers>=1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)

        if layers>=2:
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)

        if layers>=3:
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(int(length * n_filters / (2**layers)), n_neurons)
        self.fc1_drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.layers>=1:
            x = self.pool1(F.relu(self.conv1(x)))
        if self.layers>=2:
            x = self.pool2(F.relu(self.conv2(x)))
        if self.layers>=3:
            x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))

        return x

# F.gumbel_softmax(logits, tau=1, hard=True)

class multiTimeAttention(nn.Module):
    "mTAND module"
    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)

            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn=F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        # embed_dim can be decomposed into num_heads x head_dim?
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # src_len = bsz * num_heads
        src_len = k.size(1)
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                # print(attn_weights.shape)
                # print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # embed_dim = self.num_heads * self.head_dim
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, device,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False,learn_embed=True, q_seq_len=None, kv_seq_len=None,):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device=device
        self.q_seq_len=q_seq_len
        self.kv_seq_len=kv_seq_len
        if learn_embed:
            if self.q_seq_len!=None:
                self.embed_positions_q=nn.Embedding(self.q_seq_len,embed_dim,padding_idx=0)
                nn.init.normal_(self.embed_positions_q.weight, std=0.02)

            if self.kv_seq_len!=None:
                self.embed_positions_kv=nn.Embedding(self.kv_seq_len,embed_dim)
                nn.init.normal_(self.embed_positions_kv.weight, std=0.02)

        else:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        x=x_in
        length_x = x.size(0) # (length,Batch_size,input_dim)
        x = self.embed_scale * x_in
        if self.q_seq_len is not None:
            position_x = torch.tensor(torch.arange(length_x),dtype=torch.long).to(self.device)
            x += (self.embed_positions_q(position_x).unsqueeze(0)).transpose(0,1)  # Add positional embedding
        x =F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions

            length_kv = x_in_k.size(0) # (Batch_size,length,input_dim)
            position_kv = torch.tensor(torch.arange(length_kv),dtype=torch.long).to(self.device)

            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.kv_seq_len is not None:
                x_k += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
                x_v += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerCrossEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerCrossEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, args, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, q_seq_len_1=None, q_seq_len_2=None, num_modalities=2):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device=device

        self.q_seq_len_1=q_seq_len_1 
        # seq_len_1 is tt_max, the longest sequence length, which is 48 for 48 hrs
        self.q_seq_len_2=q_seq_len_2
        self.num_modalities = num_modalities
        # self.intermediate=intermediate
        self.embed_positions_q_1=nn.Embedding(self.q_seq_len_1, embed_dim, padding_idx=0)
        nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)

        if self.q_seq_len_2 != None:
            self.embed_positions_q_2=nn.Embedding(self.q_seq_len_2,embed_dim,padding_idx=0)
            nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)
            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1, self.embed_positions_q_2])
        else:
            self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1 for _ in range(num_modalities)])

        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerCrossEncoderLayer(args,
                                                    embed_dim,
                                                    num_heads=num_heads,
                                                    attn_dropout=attn_dropout,
                                                    relu_dropout=relu_dropout,
                                                    res_dropout=res_dropout,
                                                    attn_mask=attn_mask,
                                                    num_modalities=num_modalities)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])

    def forward(self, x_in_list, modality):
        """
        Args:
            x_in_list (list of FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the list of last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

        """

        # x_in_list contains ts and clinical notes tensors
        x_list = x_in_list
        lengths, positions = [], []
        for i in range(self.num_modalities):
            lengths.append(x_list[i].size(0))
        x_list = [self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            for length in lengths:
                positions.append(torch.tensor(torch.arange(length),dtype=torch.long).to(self.device))
            x_list = [l(position_x).unsqueeze(0).transpose(0,1) + x for l, x, position_x in zip(self.embed_positions_q, x_list, positions)]
              # Add positional embedding
            x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]
        # encoder layers
        for layer in self.layers:
            x_list = layer(x_list, modality) #proj_x_txt, proj_x_ts
            # len(x_list) = 2
            # x_list[0].shape = torch.Size([48, 1, 128])
            if x_list is None:
                return None

        if self.normalize:
            x_list=[l(x) for l, x in zip(self.layer_norm, x_list)]
        return x_list


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, args, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, 
                 attn_mask=False, num_modalities=2):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.self_attns = nn.ModuleList([MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        ) for _ in range(num_modalities)])

        self.post_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.cross_attn_1 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.cross_attn_2 = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.post_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        self.fc1 = nn.ModuleList([nn.Linear(self.embed_dim, 4 * self.embed_dim) for _ in range(num_modalities)])  # The "Add & Norm" part in the paper
        self.fc2 = nn.ModuleList([nn.Linear(4 * self.embed_dim, self.embed_dim) for _ in range(num_modalities)])
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
            self.moe = self.moe.to('cuda:0')
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
            self.moe = self.moe.to('cuda:0')
        
    def forward(self, x_list, modality):
        """
        Args:
            x (List of Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            list of encoded output of shape `(batch, src_len, embed_dim)`
        """
        #TODO: figure out how many layers of this is required?
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

        # pay attention to how the text and patch embeddings are concated in LIMOE
        # LIMOE just concat? add modality type embeddings
        # sparse attention combined with dense attention
        if self.args.cross_method == "self_cross":
            assert self.num_modalities == 2, 'Input modality should be 2 if using cross attention method.'
            x_txt, x_ts = x_list #proj_x_txt, proj_x_ts
            x_ts_to_txt, _ = self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
            x_txt_to_ts, _ = self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)

            x_ts_to_txt = F.dropout(x_ts_to_txt, p=self.res_dropout, training=self.training)
            x_txt_to_ts = F.dropout(x_txt_to_ts, p=self.res_dropout, training=self.training)
            x_list = [r+ x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]
        x_list = [F.dropout(x, p=self.relu_dropout, training=self.training) for x in x_list]
        x_list = [l(x) for l, x in zip(self.fc2, x_list)]
        x_list = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list]
        x_list = [r + x  for r, x in zip(residual, x_list) ]
        return x_list


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:bpbpp
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
#     nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)

    return m


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


class TransformerEncoderLayerMoE(nn.Module):
    def __init__(self, args, embed_dim, device, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, 
                 embed_dropout=0.0, attn_mask=False, num_modalities=1):
        super().__init__()
        self.device = device
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
            MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, attn_dropout=attn_dropout) 
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
            self.moe = self.moe.to(device)
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
            self.moe = self.moe.to(device)

    def forward(self, x_list):
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
            moe_out, balance_loss = self.moe(x_mod_in)
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
                 embed_dropout=0.0, attn_mask=False, q_seq_len=None, kv_seq_len=None, num_modalities=1):
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
                                                    num_modalities=1,
                                                    device=device)
            self.layers.append(new_layer)
        
        self.embed_positions = nn.Embedding(q_seq_len, embed_dim, padding_idx=0) if q_seq_len else None
        nn.init.normal_(self.embed_positions.weight, std=0.02) if q_seq_len else None
        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_list = [x]  # Ensure x is treated as a list
        lengths, positions = [], []
        lengths.append(x_list[0].size(0))
        x_list = [self.embed_scale * x_in for x_in in x_list]
        for length in lengths:
            position_x = torch.tensor(torch.arange(length), dtype=torch.long).to(self.device)
            # Directly use the embedding object without indexing
            x_list = [(self.embed_positions(position_x).unsqueeze(0).transpose(0, 1) + x_in) for x_in in x_list]
            # Add positional embedding
        x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]

        for layer in self.layers:
            x_list = layer(x_list)  # Assuming layer can handle call without modality
            if x_list is None:
                return None

        if self.normalize:
        # Apply LayerNorm directly instead of trying to iterate over it
            x_list = [self.layer_norm(x) for x in x_list]

        return x_list
