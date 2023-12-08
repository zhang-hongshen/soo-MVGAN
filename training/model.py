from typing import Tuple, List

import torch
from torch import nn

from layer import FreeEmbeddingLayer, AttributeEmbeddingLayer, EmbeddingFusionLayer


# 自由嵌入网络
class FreeEmbeddingNetwork(nn.Module):
    def __init__(self,
                 leaky_relu_negative_slope: float = 0.2,
                 dropout: float = 0.2):
        super().__init__()
        # 初始化两层自由嵌入层
        self.layer1 = FreeEmbeddingLayer()
        self.layer2 = FreeEmbeddingLayer()
        # 激活函数设置为leakyReLU
        self.activation = nn.LeakyReLU(leaky_relu_negative_slope)
        self.dropout = nn.Dropout(dropout)

    def forward(self, items, neighbors, weight, bias) -> Tuple[torch.Tensor, torch.Tensor]:
        # 第一层前向传播
        users, products = self.layer1.forward(items, neighbors, weight, bias)
        # 结果dropout处理
        items = self.dropout(self.activation(users)), self.dropout(self.activation(products))
        # 第二层前向传播
        users, products = self.layer2.forward(items, neighbors, weight, bias)
        return self.dropout(self.activation(users)), self.dropout(self.activation(products))


# 多层感知器
class MLP(nn.Module):
    def __init__(self, in_features, n_hidden, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, n_hidden)  # 隐藏层
        self.linear2 = nn.Linear(n_hidden, out_features)  # 输出层
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, din: torch.Tensor):
        dim0, dim1 = din.shape
        din = din.reshape(dim0 * dim1)
        dout = self.linear1(din)
        dout = self.relu(dout)
        dout = self.linear2(dout)
        return self.relu(dout)


# 预测器
class Predictor:
    def __init__(self):
        self.mlp = MLP(3 * 64, 100, 1).cuda()

    def predict(self, user, product):
        n_user, n_product = len(user), len(product)
        R = torch.zeros([n_user, n_product]).cuda()
        # 计算偏好分数
        for i in range(n_user):
            for j in range(n_product):
                # 合并为一个3*64的tensor
                e = torch.stack([user[i] * product[j], user[i], product[j]], dim=0)
                R[i, j] = self.mlp(e)
        return R


class MultiViewGraphAttentionNetWork(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.reset_parameters(embedding_size)
        self.free_layer = FreeEmbeddingNetwork()
        self.attribute_layer = AttributeEmbeddingLayer()
        self.fusion_layer = EmbeddingFusionLayer()

    def forward(self, items, user_product, user_attr, product_attr):

        users, products = items[0], items[1]
        view1_users, view1_products = self.free_layer([users, products],
                                                      user_product,
                                                      self.weight,
                                                      [self.user_bias, self.product_bias])
        usv = [user_attr[0], product_attr[0].T]
        uvsv = [user_product, product_attr[0], product_attr[0].T]
        usuv = [user_attr[0], user_attr[0].T, user_product]
        view2_user_neighbors = [self.metapath_guided_neighbors(usv),
                                self.metapath_guided_neighbors(uvsv),
                                self.metapath_guided_neighbors_3(usuv)]
        vsu = [product_attr[0], user_attr[0].T]
        vusu = [user_product.T, user_attr[0], user_attr[0].T]
        vsvu = [product_attr[0], product_attr[0].T, user_product.T]
        view2_product_neighbors = [self.metapath_guided_neighbors(vsu),
                                   self.metapath_guided_neighbors(vusu),
                                   self.metapath_guided_neighbors_3(vsvu)]
        view2_users, view2_products = self.attribute_layer([users, products],
                                                           [view2_user_neighbors, view2_product_neighbors],
                                                           [self.view2_user_X, self.view2_product_X],
                                                           [self.view2_user_V, self.view2_product_V],
                                                           [self.view2_user_W_p, self.view2_product_W_p],
                                                           [self.view2_user_B_p, self.view2_product_B_p],
                                                           [self.view2_user_W_q, self.view2_product_W_q],
                                                           [self.view2_user_B_q, self.view2_product_B_q],
                                                           [self.view2_user_Q, self.view2_product_Q])
        udv = [user_attr[1], product_attr[1].T]
        uvdv = [user_product, product_attr[1], product_attr[0].T]
        uduv = [user_attr[1], user_attr[1].T, user_product]
        view3_user_neighbors = [self.metapath_guided_neighbors(udv),
                                self.metapath_guided_neighbors(uvdv),
                                self.metapath_guided_neighbors_3(uduv)]
        vdu = [product_attr[1], user_attr[1].T]
        vudu = [user_product.T, user_attr[1], user_attr[1].T]
        vdvu = [product_attr[1], product_attr[1].T, user_product.T]
        view3_product_neighbors = [self.metapath_guided_neighbors(vdu),
                                   self.metapath_guided_neighbors(vudu),
                                   self.metapath_guided_neighbors_3(vdvu)]
        view3_users, view3_products = self.attribute_layer([users, products],
                                                           [view3_user_neighbors, view3_product_neighbors],
                                                           [self.view3_user_X, self.view3_product_X],
                                                           [self.view3_user_V, self.view3_product_V],
                                                           [self.view3_user_W_p, self.view3_product_W_p],
                                                           [self.view3_user_B_p, self.view3_product_B_p],
                                                           [self.view3_user_W_q, self.view3_product_W_q],
                                                           [self.view3_user_B_q, self.view3_product_B_q],
                                                           [self.view3_user_Q, self.view3_product_Q])
        uvpv = [user_product, product_attr[2], product_attr[2].T]
        view4_user_neighbors = [self.metapath_guided_neighbors(uvpv)]
        vpvu = [product_attr[2], product_attr[2].T, user_product.T]
        view4_product_neighbors = [self.metapath_guided_neighbors_3(vpvu)]
        view4_users, view4_products = self.attribute_layer([users, products],
                                                           [view4_user_neighbors, view4_product_neighbors],
                                                           [self.view4_user_X, self.view4_product_X],
                                                           [self.view4_user_V, self.view4_product_V],
                                                           [self.view4_user_W_p, self.view4_product_W_p],
                                                           [self.view4_user_B_p, self.view4_product_B_p],
                                                           [self.view4_user_W_q, self.view4_product_W_q],
                                                           [self.view4_user_B_q, self.view4_product_B_q],
                                                           [self.view4_user_Q, self.view4_product_Q])
        view_users = torch.stack([view1_users, view2_users, view3_users, view4_users], dim=0).cuda()
        view_products = torch.stack([view1_products, view2_products, view3_products, view4_products], dim=0).cuda()
        users, products = self.fusion_layer([view_users, view_products],
                                            [self.user_view_weight, self.product_view_weight])
        return users, products

    def metapath_guided_neighbors(self, matrices: List[torch.Tensor]):
        n_matrix = len(matrices)
        if n_matrix <= 0:
            return
        dim0, dim1 = matrices[0].shape
        adj_mat = torch.ones([dim0, dim0]).cuda()
        for matrix in matrices:
            adj_mat = adj_mat @ matrix
        return [[j for j, value in enumerate(row) if value > 0]
                for i, row in enumerate(adj_mat)]

    def metapath_guided_neighbors_3(self, matrices):
        adj_mat = matrices[0] @ matrices[1]
        adj_mat = adj_mat - torch.diag_embed(torch.diag(adj_mat))
        adj_mat = adj_mat @ matrices[2]

        return [[j for j, value in enumerate(row) if value > 0]
                for i, row in enumerate(adj_mat)]

    def reset_parameters(self, embedding_size, d_p=3, d_q=4):
        # 参数初始化
        # ID嵌入层参数初始化
        self.weight = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([embedding_size, embedding_size], requires_grad=True)))
        self.user_bias = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, embedding_size], requires_grad=True)))
        self.product_bias = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, embedding_size], requires_grad=True)))
        # 属性嵌入层参数初始化
        self.view2_user_X = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view2_user_V = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view2_user_W_p = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view2_user_B_p = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view2_user_W_q = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_q], requires_grad=True)))
        self.view2_user_B_q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view2_user_Q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view3_user_X = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view3_user_V = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view3_user_W_p = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view3_user_B_p = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view3_user_W_q = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_q], requires_grad=True)))
        self.view3_user_B_q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view3_user_Q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view4_user_X = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_p], requires_grad=True)))
        self.view4_user_V = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([1, embedding_size, d_p], requires_grad=True)))
        self.view4_user_W_p = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([1, embedding_size, d_p], requires_grad=True)))
        self.view4_user_B_p = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_p], requires_grad=True)))
        self.view4_user_W_q = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([1, embedding_size, d_q], requires_grad=True)))
        self.view4_user_B_q = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_q], requires_grad=True)))
        self.view4_user_Q = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_q], requires_grad=True)))

        self.view2_product_X = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view2_product_V = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view2_product_W_p = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view2_product_B_p = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view2_product_W_q = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_q], requires_grad=True)))
        self.view2_product_B_q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view2_product_Q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view3_product_X = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view3_product_V = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view3_product_W_p = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_p], requires_grad=True)))
        self.view3_product_B_p = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_p], requires_grad=True)))
        self.view3_product_W_q = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([3, embedding_size, d_q], requires_grad=True)))
        self.view3_product_B_q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view3_product_Q = nn.Parameter(nn.init.xavier_normal_(torch.empty([3, 1, d_q], requires_grad=True)))
        self.view4_product_X = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_p], requires_grad=True)))
        self.view4_product_V = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([1, embedding_size, d_p], requires_grad=True)))
        self.view4_product_W_p = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([1, embedding_size, d_p], requires_grad=True)))
        self.view4_product_B_p = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_p], requires_grad=True)))
        self.view4_product_W_q = nn.Parameter(
            nn.init.xavier_normal_(torch.empty([1, embedding_size, d_q], requires_grad=True)))
        self.view4_product_B_q = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_q], requires_grad=True)))
        self.view4_product_Q = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 1, d_q], requires_grad=True)))
        # 嵌入融合层参数初始化
        self.user_view_weight = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 4], requires_grad=True)))
        self.product_view_weight = nn.Parameter(nn.init.xavier_normal_(torch.empty([1, 4], requires_grad=True)))
