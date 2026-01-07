import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
import torch.nn.functional as F
from torch.distributed import group


class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        B,T,_,D = x.shape
        # P: ptach num and N: patch size
        P, N = self.num, self.size
        assert self.num * self.size == _
        x = x.reshape(B, T, P, N, D)

        # depth attention
        qkv = self.snorm1(x.reshape(B*T*P,N,D))
        x = x + self.sattn(qkv).reshape(B,T,P,N,D)
        x = x + self.smlp(self.snorm2(x))

        # breadth attention
        qkv = self.nnorm1(x.transpose(2,3).reshape(B*T*N,P,D))
        x = x + self.nattn(qkv).reshape(B,T,N,P,D).transpose(2,3)
        x = x + self.nmlp(self.nnorm2(x))

        return x.reshape(B,T,-1,D)

class PatchSTG(nn.Module):
    def __init__(self, output_len, tem_patchsize, tem_patchnum,
                        node_num, spa_patchsize, spa_patchnum,
                        tod, dow,
                        layers, factors,
                        input_dims, node_dims, tod_dims, dow_dims,
                        ori_parts_idx, reo_parts_idx, reo_all_idx
                ):
        super(PatchSTG, self).__init__()
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow

        # model_dims = input_emb + spa_emb + tem_emb
        dims = input_dims + tod_dims + dow_dims + node_dims

        # spatio-temporal embedding -> section 4.1 in paper
        # input_emb
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        # spa_emb
        self.node_emb = nn.Parameter(
                torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        # tem_emb
        self.time_in_day_emb = nn.Parameter(
                torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # dual attention encoder -> section 4.3 in paper, factors for merging the leaf nodes of KDTree
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, spa_patchnum//factors, spa_patchsize*factors, mlp_ratio=1) for _ in range(layers)
        ])
        print(f'spa_patchnum//factors:{spa_patchnum//factors}, spa_patchsize*factors{spa_patchsize*factors}')

        # projection decoder -> section 4.4 in paper
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims, out_channels=output_len, kernel_size=(1, 1), bias=True)

    def forward(self, x, te):
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information

        # spatio-temporal embedding -> section 4.1 in paper
        embeded_x = self.embedding(x, te)
        rex = embeded_x[:,:,self.reo_all_idx,:] # select patched points

        # dual attention encoder -> section 4.3 in paper
        for block in self.spa_encoder:
            rex = block(rex)

        orginal = torch.zeros(rex.shape[0],rex.shape[1],self.node_num,rex.shape[-1]).to(x.device)
        orginal[:,:,self.ori_parts_idx,:] = rex[:,:,self.reo_parts_idx,:] # back to the original indices

        # projection decoder -> section 4.4 in paper
        print(f'orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1) shape:{orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1).shape}')
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y # [B,T,N,1]

    def embedding(self, x, te):
        b,t,n,_ = x.shape

        # input traffic + time of day + day of week as the input signal
        x1 = torch.cat([x,(te[...,0:1]/self.tod),(te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]

        # cat time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # cat day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # cat spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data
class MySTG(nn.Module):
    def __init__(self, output_len, tem_patchsize, tem_patchnum,
                        node_num, group_size, group_num,
                        tod, dow,
                        layers,
                        input_dims, node_dims, tod_dims, dow_dims, dropout, ff_dim, group_matrix
                ):
        super(MySTG, self).__init__()
        self.node_num = node_num
        self.tod, self.dow = tod, dow
        self.group_size = group_size
        self.group_num = group_num
        self.node_dims = node_dims

        # model_dims = input_emb + spa_emb + tem_emb
        dims = input_dims + tod_dims + dow_dims + node_dims

        # spatio-temporal embedding -> section 4.1 in paper
        # input_emb
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        # spa_emb
        self.node_emb = nn.Parameter(
                torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.group_emb = nn.Parameter(
            torch.empty(group_num, node_dims))
        nn.init.xavier_uniform_(self.group_emb)
        # tem_emb
        self.time_in_day_emb = nn.Parameter(
                torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.group_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dims, nhead=4, dim_feedforward=ff_dim, dropout=dropout, batch_first=True),
            layers
        )
        # self.member_transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(dims, nhead=4, dim_feedforward=ff_dim, dropout=dropout, batch_first=True),
        #     layers)
        self.member_transformer_layers = nn.ModuleList([
            CrossAttentionBlock(d_model=dims, nhead=4, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        # projection decoder -> section 4.4 in paper
        #self.regression = nn.Linear(dims*2, dims)
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum * dims, out_channels=output_len, kernel_size=(1, 1),
                                         bias=True)

        # self.gcn_weight = nn.Parameter(torch.Tensor(dims, dims))
        # nn.init.xavier_uniform_(self.gcn_weight)
        # self.member_mixer = nn.Sequential(
        #     nn.Linear(group_num, group_num*4),
        #     nn.GELU(),
        #     nn.Linear(group_num*4, group_num)
        # )
        # self.node_to_q = nn.Linear(node_dims, node_dims, bias=False) # 投影到匹配空间
        # self.group_to_k = nn.Linear(node_dims, node_dims, bias=False)
        # self.group_matrix = nn.Parameter(torch.empty(node_num, group_num))
        # numpy_array = np.loadtxt('models/G_ca.csv', delimiter=',', dtype=np.float32)
        #self.group_pos_emb = nn.Parameter(torch.randn(1, group_num, dims))
        # if group_matrix is not None:
        #     self.group_matrix =  nn.Parameter(group_matrix, requires_grad=True)
        # else:
        #     self.group_matrix = nn.Parameter(torch.empty(node_num, group_num))
        #     nn.init.xavier_normal_(self.group_matrix)
        # self.group_learner = nn.Sequential(
        #     nn.Linear(dims, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, group_num)
        # )
        # self.res_mlp = nn.Sequential(nn.Linear(dims, dims//2),
        #                              nn.ReLU(),
        #                              nn.Linear(dims//2, dims))
        # nn.init.xavier_normal_(self.group_matrix)

    def forward(self, x, te):
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information

        # spatio-temporal embedding -> section 4.1 in paper

        # embedded_x: [B,1,N,D] input traffic
        #group_matrix = self.group_matrix #group_matrix: [node_num, group_num]
        #G = F.softmax(self.group_matrix, dim=0)
        # node_q: [N, 64]
        #node_q = self.node_to_q(self.node_emb)
        # group_k: [G, 64]
        #group_k = self.group_to_k(self.group_emb)
        #group_matrix = node_q @ self.group_emb.transpose(0, 1)
        #group_matrix = group_matrix / math.sqrt(self.node_dims)
        group_matrix = self.node_emb @ self.group_emb.transpose(0, 1)
        G=F.softmax(group_matrix, dim=-1)

        group_indices = torch.argmax(G, dim=1)  # (N,) recording nodes belongs to which group

        index = G.max(dim=-1, keepdim=True)[1]
        probs_hard = torch.zeros_like(group_matrix).scatter_(-1, index, 1.0)
        probs = probs_hard - group_matrix.detach() + group_matrix

        # 2. 批量处理：按分组索引排序，然后批量处理
        sorted_indices = torch.argsort(group_indices)
        sorted_group_indices = group_indices[sorted_indices] # recording sorted group indices

        group_emb = self.group_emb.gather(
            0,
            sorted_group_indices.unsqueeze(-1).expand(-1, self.node_dims)
        )
        node_emb = self.node_emb + group_emb

        embedded_x = self.embedding(x, te, node_emb)
        batch_size, _, num_nodes, dim = embedded_x.shape

        group_x = G.transpose(0,1) @ embedded_x

        # graph = torch.matmul(self.group_emb, self.group_emb.transpose(0, 1))
        # group_graph = F.softmax(F.relu(graph), dim=-1)
        #
        # support = group_x.squeeze(1) @ self.gcn_weight
        # gcn_out = group_graph @ support
        # y = gcn_out.transpose(1, 2)
        # y = self.member_mixer(y)
        # group_out = y.transpose(1, 2)

        group_out = self.group_transformer(group_x.squeeze(1)) #[B, g, D]
        group_out_G = G @ group_out
        group_out_res = group_out_G+embedded_x.squeeze(1) #[B, N, D]
        # mlp_out = self.res_mlp(group_out+embedded_x.squeeze(1))
        # mlp_out = mlp_out.transpose(1,2).unsqueeze(-1)

        sorted_embeddings = group_out_res[:,sorted_indices,:]

        # 3. 找到每个分组的边界
        group_boundaries = torch.cat([
            torch.tensor([0], device=group_out_res.device),
            torch.where(torch.diff(sorted_group_indices) != 0)[0] + 1,
            torch.tensor([num_nodes], device=group_out_res.device)
        ])
        group_context = group_x.squeeze(1).gather(
            1,
            sorted_group_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, dim)
        )  # [B, N, D]

        # 4. 批量处理每个分组（避免小矩阵操作）
        processed_embeddings = torch.zeros_like(sorted_embeddings)
        for i in range(len(group_boundaries) - 1):
            start, end = group_boundaries[i], group_boundaries[i+1]
            group_size = end - start

            if group_size > 0:
                # 批量处理整个分组
                group_emb = sorted_embeddings[:, start:end, :]
                context = group_context[:, start:end, :]

                # Transformer处理
                #processed_group = self.member_transformer(group_emb)
                for layer in self.member_transformer_layers:
                    group_emb = layer(group_emb, context)
                processed_embeddings[:, start:end, :] = group_emb

        # 5. 还原原始顺序
        reverse_indices = torch.argsort(sorted_indices)
        result = processed_embeddings[:, reverse_indices, :].transpose(1, 2).unsqueeze(-1)

        pred_y = self.regression_conv(result)

        return pred_y, probs # [B,T,N,1]

    def embedding(self, x, te, node_emb):
        b,t,n,_ = x.shape

        # input traffic + time of day + day of week as the input signal
        x1 = torch.cat([x,(te[...,0:1]/self.tod),(te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]

        # cat time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # cat day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # cat spatial embedding
        node_emb = node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super().__init__()
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Cross-Attention: Query = nodes, Key/Value = group context
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, context, src_mask=None, src_key_padding_mask=None):
        """
        x: [B, N, D]  node features
        context: [B, N, D]  group context for each node
        """
        x_ori = x
        x2 = self.norm1(x)
        x2, _ = self.self_attn(
            context, x2, x2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout1(x2)

        # Cross-Attention: Query=x, Key=Value=context
        x2 = self.norm2(x)
        x2, _ = self.cross_attn(
            x_ori, x2, x2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout2(x2)

        # FFN
        x2 = self.norm3(x)
        x2 = self.linear2(self.dropout(F.gelu(self.linear1(x2))))
        x = x + self.dropout3(x2)
        # # Self-Attention
        # x2, _ = self.self_attn(
        #     context, x, x,
        #     attn_mask=src_mask,
        #     key_padding_mask=src_key_padding_mask
        # )
        # # x = x + self.dropout1(x2)
        # g = torch.cat([x2, x], dim=1)
        # # Cross-Attention: Query=x, Key=Value=context
        # x2, _ = self.cross_attn(
        #     x, g, g,
        #     attn_mask=src_mask,
        #     key_padding_mask=src_key_padding_mask
        # )
        # x = x + self.dropout2(x2)
        #
        # # FFN
        # x2 = self.norm1(x)
        # x2 = self.linear2(self.dropout(F.gelu(self.linear1(x2))))
        # x = x + self.dropout3(x2)
        # x = self.norm2(x)

        return x

