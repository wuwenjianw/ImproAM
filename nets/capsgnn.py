import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch_geometric.nn as pyg
from torch_geometric.nn.inits import glorot, zeros
from utils.functions import move_to
import math


class linearDisentangle(torch.nn.Module):
    """线性层"""
    def __init__(self, in_dims, out_dims):
        super(linearDisentangle, self).__init__()
        self.weight = Parameter(torch.Tensor(in_dims, out_dims))
        self.bias = Parameter(torch.Tensor(out_dims))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        # self.weight2 = Parameter(torch.Tensor(self.out_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        # glorot(self.weight2)
        zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop: # 添加自环
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2
        out = torch.matmul(x, self.weight)  # [16, 28, 36]

        # 邻接矩阵归一化
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)  # [16, 28, 36] 将归一化后的邻接矩阵adj与变换后的特征矩阵 out 相乘


        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Attention(nn.Module):
    def __init__(self, input_dim,output_dim, hidden_dim=None,  num_layers=2, activation=torch.tanh):
        super(Attention, self).__init__()

        self.num_layers = num_layers
        self.activation = activation
        self.linears = torch.nn.ModuleList()
        # self.batch_norms = torch.nn.ModuleList()

        if hidden_dim is None:
            hidden_dim = int(input_dim / 16) + 4

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linears.append(nn.Linear(input_dim, output_dim, bias=False))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

            # for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            # h = self.batch_norms[layer](h)
            h = self.activation(h)

        return self.linears[self.num_layers - 1](h)



def squash(input_tensor, dim=-1, epsilon=1e-11):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (1 + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector, scale


def sparse2dense(x, new_size, mask):
    out = move_to(torch.zeros(new_size), x.device)# cuda_device(torch.zeros(new_size))
    out[mask] = x
    return out


def routing(u_hat, num_iteration, mask=None):
    u_hat_size = u_hat.size()
    b_ij = torch.zeros(u_hat_size[0], u_hat_size[1], u_hat_size[2], 1, 1, device=u_hat.device)
    # _, b_ij = squash(u_hat, dim=-2)
    # b_ij = cuda_device(b_ij)  # [bs,n,upper_caps,1]

    for i in range(num_iteration - 1):
        c_ij = F.softmax(b_ij, dim=2)  # [bs,n*(neighbor+1),upper_caps,1,1]
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)  # [bs,1,upper_caps,d,1,1]
        v, a_j = squash(s_j, dim=-2)
        u_produce_v = torch.matmul(u_hat.transpose(-1, -2), v)  # [bs,n*(neighbor+1),upper_caps,1,1]
        b_ij = b_ij + u_produce_v  # [bs,n*(neighbor+1),upper_caps,1,1]
    if mask is not None:
        b_ij = b_ij * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, b_ij.size(2),
                                                                            b_ij.size(3),
                                                                            b_ij.size(4))
    c_ij = F.softmax(b_ij, dim=2)  # [bs,n*(neighbor+1),upper_caps,1,1]
    # c_ij = F.sigmoid(b_ij)  # [bs,n*(neighbor+1),upper_caps,1,1]
    return c_ij


class firstCapsuleLayer(torch.nn.Module):
    def __init__(self, number_of_features, features_dimensions, capsule_dimensions, num_gcn_layers, num_gcn_channels,
                dropout):
        super(firstCapsuleLayer, self).__init__()

        self.number_of_features = number_of_features
        self.features_dimensions = features_dimensions
        self.capsule_dimensions = capsule_dimensions
        self.dropout = nn.Dropout(p=dropout)
        self.num_gcn_layers = num_gcn_layers
        self.num_gcn_channels = num_gcn_channels
        self.gcn_layers_dims = self.capsule_dimensions ** 2

        # self.bn = nn.BatchNorm1d(self.number_of_features)
        # print(self.number_of_features)
        # 
        self.encapsule = nn.ModuleList()
        for i in range(self.capsule_dimensions):
            self.encapsule.append(
                linearDisentangle(self.number_of_features, self.capsule_dimensions))
        # GCN
        self.gcn_layers = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            self.gcn_layers.append(DenseGCNConv(self.gcn_layers_dims, self.gcn_layers_dims))

        self.attention = Attention(self.gcn_layers_dims * self.num_gcn_layers, self.num_gcn_layers)

    def forward(self, x, adj, mask):

        x_size = x.size()  # [16, 100, 4]  100表示图的节点数，4表示节点的特征数
        x = x[mask]  # (N1+N2+...+Nm)*d  [280, 19]  # 使用mask过滤节点，得到有效的节点特征
        out = []
        
        # 计算得到6个初始胶囊层
        for i, encaps in enumerate(self.encapsule):
            temp = F.relu(encaps(x))
            # temp = self.dropout(temp)
            out.append(temp)

        out = torch.cat(out, dim=-1)  # [280, 36]  将所有胶囊层的输出在特征维度上进行拼接

        # !如果是拼接所有胶囊层的输出，是没办法泛化到任何数量的节点！

        out = sparse2dense(out, (x_size[0], x_size[1], out.size(-1)), mask)  # [16, 28, 36] 稀疏到密集转换
        out, _ = squash(out)  # 非线性激活
        features = out

        hidden_representations = []
        for layer in self.gcn_layers:
            features = layer(features, adj, mask)  # []
            features = torch.tanh(features)
            features = self.dropout(features)
            hidden_representations.append(features.reshape(x_size[0], x_size[1], 1, -1))
        hidden_representations = torch.cat(hidden_representations, dim=2)  # [16, 28, 3, 36] 在新维度中拼接所有图卷积层的输出，形成一个四维张量

        # attention模块(可堆叠的多层线性变换模块)的输入数据是三维数据，形状为[16, 28, 108]
        attn = self.attention(hidden_representations.reshape(x_size[0], x_size[1], -1))  # 输出为[16, 28, 3]
        attn = F.softmax(attn.masked_fill(mask.unsqueeze(-1).eq(0), 0), dim=-1)  # 输出为[16, 28, 3]
        # attn1 = attn.masked_fill(mask.unsqueeze(-1).eq(0), 0)
        # attn1 = attn1.unsqueeze(-1)
        # number_of_nodes = torch.sum(mask, dim=1, keepdim=True).float().unsqueeze(-1)
        # # hidden_representations = hidden_representations * attn * number_of_nodes
        hidden_representations = hidden_representations * attn.unsqueeze(-1)  # [16, 28, 3, 36]
        return hidden_representations.reshape(x_size[0], -1, self.gcn_layers_dims)  # 最终的返回为[16, 84, 36]


# DR
class SecondaryCapsuleLayer(torch.nn.Module):
    def __init__(self, k, batch_size, num_iterations, num_capsules, low_num_capsules, in_cap_dim, out_cap_dim,
                num_gcn_layers,
                dropout):
        super(SecondaryCapsuleLayer, self).__init__()
        self.num_iterations = num_iterations  # 胶囊之间迭代次数
        self.num_higher_cap = num_capsules  # 高层胶囊的数量
        self.num_lower_cap = low_num_capsules  # 低层胶囊的数量
        self.num_gcn_layers = num_gcn_layers  # 图卷积层数量
        self.in_cap_dim = in_cap_dim
        self.out_cap_dim = out_cap_dim
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(self.in_cap_dim)
        self.k = k
        self.batch_size = batch_size
        # self.W = torch.nn.Parameter(
        #     torch.randn(self.num_lower_cap, self.num_higher_cap, self.in_cap_dim, self.out_cap_dim))
        self.W = torch.nn.Parameter(torch.randn(self.k, self.in_cap_dim, self.out_cap_dim))
        self.alpha = torch.nn.Parameter(torch.randn(self.num_lower_cap, self.num_higher_cap, self.k))
        # self.gcn = DenseGCNConv(self.in_cap_dim ** 2, self.out_cap_dim ** 2)

    def forward(self, x, adj, mask=None):

        x_size, batch_size, max_node_num = x.size(), x.size(0), x.size(1)  # [b, max_node_num, in_cap_dim]
        if mask is not None:
            mask = mask.repeat(1, self.num_gcn_layers)  # [b, max_node_num] 扩展其维度以适配后续层的数量

        # [num_lower_cap, num_higher_cap, in_cap_dim] [84, 10, 5]
        alpha = self.alpha  # 其中84表示低层胶囊数量，10表示高层胶囊数量，5是每个胶囊的特征维度

        # self.W 维度为 [num_lower_cap, num_higher_cap, in_cap_dim, out_cap_dim]

        # [1, num_lower_cap, num_higher_cap, out_cap_dim, in_cap_dim, in_cap_dim] [1, 84, 10, 5, 6, 6]
        tmp1 = self.W.unsqueeze(0).unsqueeze(0).unsqueeze(0) * alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # 
        tmp2  = tmp1.sum(dim=3)  # [1, 84, 10, 6, 6] 按第3个维度进行求和

        # [b, num_lower_cap, 1, in_cap_dim, in_cap_dim] [16, 84, 1, 6, 6]
        x = x.view(batch_size, self.num_lower_cap, 1, self.in_cap_dim, -1)

        # [b, num_lower_cap, num_higher_cap, in_cap_dim, in_cap_dim] [16, 84, 10, 6, 6]
        tmp4 = torch.matmul(x, tmp2)  # 计算的胶囊输出

        # [b, num_lower_cap, num_higher_cap, in_cap_dim**2, 1] [16, 84, 10, 36, 1]
        u_hat = tmp4
        u_hat = u_hat.view(batch_size, max_node_num, self.num_higher_cap, -1, 1)
        
        if mask is not None:
            # [b, num_lower_cap, num_higher_cap, in_cap_dim**2, 1] [16, 84, 10, 36, 1]
            u_hat = u_hat * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, u_hat.size(2),
                                                                                u_hat.size(3),
                                                                                u_hat.size(4))  
        
        temp_u_hat = u_hat.detach()

        # 路由操作
        # [b, num_lower_cap, num_higher_cap, 1, 1] [16, 84, 10, 1, 1]
        c_ij = routing(temp_u_hat, num_iteration=self.num_iterations, mask=mask)  

        #   [b, 1, num_higher_cap, in_cap_dim**2, 1] [16, 1, 10, 36, 1]
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
        v, a_j = squash(s_j, dim=-2)  # [16, 1, 10, 36, 1]； [16, 1, 10, 1, 1]

        return v, a_j


class CapsGNN(torch.nn.Module):
    def __init__(self, args, number_of_features):
        super(CapsGNN, self).__init__()
        self.args = args
        self.max_node_num =  args.graph_size  # max_node_num
        self.number_of_features = number_of_features

        self.init_embed = nn.Linear(3, 128 * 3)  # 将节点维度从100变为128*3
        self.activ = nn.LeakyReLU()  # 使用 LeakyReLU 激活函数
        # 用于处理图的不同变换
        self.W_L_1_G1 = nn.Linear(128 * (2 + 1) * 3, 128)
        self.W_L_1_G2 = nn.Linear(128 * (2 + 1) * 3, 128)
        self.W_L_1_G3 = nn.Linear(128 * (2 + 1) * 3, 128)
        self.W_F = nn.Linear(128 * 3, 128)  # ! 此次有修改 最后一层全连接层
        self.normalization_1 = Normalization(128 * 3)  # 使用BatchNorm1d归一化
        self.normalization_2 = Normalization(128)

        # 最后一层投影层
        self.init_embed_depot = nn.Linear(2, 128)
        # ① 特征映射：36  → 128*3
        self.feat_proj = nn.Linear(self.args.capsule_dimensions ** 2, 128 * 3) #!  此处有修改 ，可选 128*3 or 128
        # ② 时序映射：10 → 100  
        self.seq_proj  = nn.Linear(10, self.max_node_num) 
        self.p_1 = nn.Linear(128, 128 * 3) 
        self.p_2 = nn.Linear(128 * 2, 128 * 3)
        self._setup_layers()
        


    def _setup_layers(self):
        self._setup_firstCapsuleLayer()
        self._setup_hiddenCapsuleLayer()


    def _setup_firstCapsuleLayer(self):
        self.first_capsule = firstCapsuleLayer(number_of_features=self.number_of_features,
                                            features_dimensions=self.args.features_dimensions,
                                            capsule_dimensions=self.args.capsule_dimensions,
                                            num_gcn_layers=self.args.num_gcn_layers,
                                            num_gcn_channels=self.args.num_gcn_channels,
                                            dropout=self.args.dropout)

    def _setup_hiddenCapsuleLayer(self):
        self.hidden_capsule = SecondaryCapsuleLayer(k=self.args.k,
                                                    batch_size=self.args.batch_size,
                                                    num_iterations=self.args.num_iterations,
                                                    low_num_capsules=self.max_node_num * self.args.num_gcn_layers,
                                                    num_capsules=self.args.capsule_num,
                                                    in_cap_dim=self.args.capsule_dimensions,
                                                    out_cap_dim=self.args.capsule_dimensions,
                                                    num_gcn_layers=self.args.num_gcn_layers,
                                                    dropout=self.args.dropout)


    def forward(self, x):
        if True: 
            "True表示total结果，False表示单独global结果"
            # x表示特征图，adj是邻接矩阵，mask是掩码矩阵
            epsilon = 1e-7
            batch_size = x['loc'].size(0)
            node_size = x['loc'].size(1)

            feature = torch.cat([x['loc'], x['deadline'].unsqueeze(-1), x['workload'].unsqueeze(-1)], dim=-1)

            # 重新计算邻接矩阵
            # adj = torch.zeros(batch_size, node_size, node_size, device=x['loc'].device) 
            X = torch.cat((x['loc'], x['deadline'][:, :, None]), -1)
            # 将工作量归一化
            X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1) # 【2，100，3】
            
            X_loc = X
            # 计算所有节点之间的欧几里得距离,维度为【2，100，100】
            distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
            num_samples, num_locations, _ = X.size()
            # 使用欧几里得距离的倒数构建一个邻接矩阵 A，表示节点之间的连接强度
            A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
                (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
            A[A != A] = 0
            adj = A / A.max()  # 【2，100，100】


            mask = torch.ones(batch_size, node_size, dtype=torch.bool, device=x['loc'].device)
            # first_out: [16, 84, 36]
            first_out = self.first_capsule(feature, adj, mask)

            # ! 修改所有网络的max_node_num
            self.max_node_num = node_size
            self.hidden_capsule.max_node_num = node_size

            second_out, second_adj = self.hidden_capsule(first_out, adj, mask)
            second_out = second_out.squeeze(4).squeeze(1)  # [b, num_higher_cap, d] [16, 10, 36]

            # x: [B, 10, 36]  
            second_out = self.feat_proj(second_out)          # [B, 10, 128]  
            # 把长度维挪到最后，让 Linear 作用在它上面  
            second_out = second_out.transpose(1, 2)          # [B, 128, 10]  
            second_out = self.seq_proj(second_out)           # [B, 128, 100]  
            # 再换回想要的顺序  
            second_out = self.activ(second_out.transpose(1, 2))          # ![B, 100, 128] 

            # 计算度矩阵 D 【2，100，100】
            D = torch.mul(torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
                        (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
            # 第一层图卷积计算
            # Layer 1，init_embed为一个全连接层，输出 128*3
            F0 = self.init_embed(X)  # 【2，100，128*3】

            F0_squared = torch.mul(F0[:, :, :], F0[:, :, :])  # 计算平方
            F0_cube = torch.mul(F0[:, :, :], F0_squared[:, :, :])  # 计算立方
            # 计算图的拉普拉斯矩阵 L  K = 3
            L = D - A
            L_squared = torch.matmul(L, L)  # 计算拉普拉斯矩阵的平方

            if self.args.p == 1:
                # 图卷积运算，输入1152，输出128，最终维度为【2，100，128】
                g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                                torch.matmul(L, F0)[:, :, :],
                                                torch.matmul(L_squared, F0)[:, :, :]
                                                ),
                                                -1))
                # 合并和激活
                F1 = self.p_1(g_L1_1)

            elif self.args.p == 2:
                # 图卷积运算，输入1152，输出128，最终维度为【2，100，128】
                g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                                torch.matmul(L, F0)[:, :, :],
                                                torch.matmul(L_squared, F0)[:, :, :]
                                                ),
                                                -1))
                g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                                torch.matmul(L, F0_squared)[:, :, :],
                                                torch.matmul(L_squared, F0_squared)[:, :, :]
                                                ),
                                                -1))
                # 合并和激活
                F1 = self.p_2(torch.cat((g_L1_1, g_L1_2), -1))

            else:
                # 图卷积运算，输入1152，输出128，最终维度为【2，100，128】
                g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                                torch.matmul(L, F0)[:, :, :],
                                                torch.matmul(L_squared, F0)[:, :, :]
                                                ),
                                                -1))
                g_L1_2 = self.W_L_1_G2(torch.cat((F0_squared[:, :, :],
                                                torch.matmul(L, F0_squared)[:, :, :],
                                                torch.matmul(L_squared, F0_squared)[:, :, :]
                                                ),
                                                -1))
                g_L1_3 = self.W_L_1_G3(torch.cat((F0_cube[:, :, :],
                                                torch.matmul(L, F0_cube)[:, :, :],
                                                torch.matmul(L_squared, F0_cube)[:, :, :]
                                                ),
                                                -1))

                # ============ old 前融合 ==================
                # 合并和激活
                F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3), -1)


            F1 = F0 + second_out + self.activ(F1) #   +  加入原始特征
            F1 = self.normalization_1(F1)  # 使用BatchNorm1d归一化，【2，100，384】
            # 最终嵌入【2，100，128】
            F_final = self.activ(self.W_F(F1))


            # ============ new 后融合 ==================
            # F1 = torch.cat((g_L1_1, g_L1_2, g_L1_3), -1)
            # F1 = self.activ(F1) + F0   # 加入原始特征
            # F1 = self.normalization_1(F1)  # 使用BatchNorm1d归一化，【2，100，384】
            # F2 = self.normalization_2(second_out.contiguous())
            # # 最终嵌入【2，100，128】
            # F_final = (self.activ(self.W_F(F1)) + F2) / 2
            # ============ end ==================

            # # Depot嵌入【2，1，128】
            init_depot_embed = self.init_embed_depot(x['depot'])
            h = torch.cat((init_depot_embed, F_final), 1)  # 维度【2，101，128】

        else:
            # x表示特征图，adj是邻接矩阵，mask是掩码矩阵
            batch_size = x['loc'].size(0)
            node_size = x['loc'].size(1)

            feature = torch.cat([x['loc'], x['deadline'].unsqueeze(-1), x['workload'].unsqueeze(-1)], dim=-1)

            # 重新计算邻接矩阵
            # adj = torch.zeros(batch_size, node_size, node_size, device=x['loc'].device) 
            X = torch.cat((x['loc'], x['deadline'][:, :, None]), -1)
            # 将工作量归一化
            X = torch.cat((X[:, :, 0:2], (X[:, :, 2] / X[:, :, 2].max())[:, :, None]), -1) # 【2，100，3】
            
            X_loc = X
            # 计算所有节点之间的欧几里得距离,维度为【2，100，100】
            distance_matrix = (((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5
            num_samples, num_locations, _ = X.size()
            # 使用欧几里得距离的倒数构建一个邻接矩阵 A，表示节点之间的连接强度
            A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
                (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
            A[A != A] = 0
            adj = A / A.max()  # 【2，100，100】


            mask = torch.ones(batch_size, node_size, dtype=torch.bool, device=x['loc'].device)
            # first_out: [16, 84, 36]
            first_out = self.first_capsule(feature, adj, mask)

            # ! 修改所有网络的max_node_num
            self.max_node_num = node_size
            self.hidden_capsule.max_node_num = node_size

            second_out, second_adj = self.hidden_capsule(first_out, adj, mask)
            second_out = second_out.squeeze(4).squeeze(1)  # [b, num_higher_cap, d] [16, 10, 36]

            # x: [B, 10, 36]  
            second_out = self.feat_proj(second_out)          # [B, 10, 128]  
            # 把长度维挪到最后，让 Linear 作用在它上面  
            second_out = second_out.transpose(1, 2)          # [B, 128, 10]  
            second_out = self.seq_proj(second_out)           # [B, 128, 100]  
            # 再换回想要的顺序  
            second_out = self.activ(second_out.transpose(1, 2))          # ![B, 100, 128] 

            # ============ old 前融合 ==================
            # 合并和激活
            F1 = second_out # 
            # 最终嵌入【2，100，128】
            F_final = self.activ(self.W_F(F1)) # 

            # # Depot嵌入【2，1，128】
            init_depot_embed = self.init_embed_depot(x['depot'])
            h = torch.cat((init_depot_embed, F_final), 1)  # 维度【2，101，128】

        return h, None


class Normalization(nn.Module):  # 归一化层

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)  # affine=True 表示归一化层会有可学习的仿射参数（即缩放因子和偏置）

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input