from typing import Tuple
import torch
from torch import nn


# 自由嵌入层
class FreeEmbeddingLayer(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, items, user_product, weight, bias) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取初始embedding
        user, product = items[0], items[1]
        # 获取用户点击的商品
        n_product_clicked = torch.sum(user_product, dim=1).unsqueeze(1)
        # 获取点击商品的用户
        n_user_click = torch.sum(user_product.T, dim=1).unsqueeze(1)
        # 将大于1的点击次数变成1次，模型不关注点击次数
        n_product_clicked = torch.where(n_product_clicked == 0, torch.ones_like(n_product_clicked), n_product_clicked)
        n_user_click = torch.where(n_user_click == 0, torch.ones_like(n_user_click), n_user_click)
        # 根据模型规则进行更新
        product_clicked_embedding = (user_product @ product) / n_product_clicked
        user_click_embedding = (user_product.T @ user) / n_user_click
        user = (user + product_clicked_embedding + bias[0]) @ weight
        product = (product + user_click_embedding + bias[1]) @ weight
        return user, product


# 属性嵌入层
class AttributeEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, items, metapath_guided_neighbors,
                X, V, W_p, B_p, W_q, B_q, Q) -> Tuple[torch.Tensor, torch.Tensor]:
        # 公共参数
        user, product = items[0], items[1]
        # 元路径引导的邻居
        user_metapath_guided_neighbors = metapath_guided_neighbors[0]
        # 元路径个数
        n_user_metapath = len(user_metapath_guided_neighbors)
        n_user, embedding_size = user.shape
        # 产品个数
        n_product = len(product)
        # 用户可训练参数
        user_A = torch.full([n_user_metapath, n_user, n_product], -1e-9).cuda()
        user_H = torch.zeros([n_user_metapath, n_user, embedding_size]).cuda()
        user_Beta = torch.zeros(n_user_metapath).cuda()
        user_V, user_X, user_W_p = V[0], X[0], W_p[0]
        user_B_p, user_W_q, user_B_q = B_p[0], W_q[0], B_q[0]
        user_Q = Q[0]
        for i in range(n_user_metapath):
            user_v_p, user_x_p = user_V[i], user_X[i]
            user_w_p, user_b_p = user_W_p[i], user_B_p[i]
            for j in range(n_user):
                for k in user_metapath_guided_neighbors[i][j]:
                    user_A[i, j, k] = torch.tanh(
                        (user[j] @ user_v_p + product[k] @ user_w_p + user_b_p)) @ user_x_p.T
            user_A = torch.softmax(user_A, dim=2)
            for j in range(n_user):
                user_H[i, j] = user_H[i, j] + user[j]
                for k in user_metapath_guided_neighbors[i][j]:
                    user_H[i, j] = user_H[i, j] + user_A[i, j, k].data * product[k]
            user_w_q, user_b_q, user_q = user_W_q[i], user_B_q[i], user_Q[i]
            for j in range(n_user):
                user_Beta[i] = user_Beta[i] + (torch.tanh(user_H[i, j] @ user_w_q + user_b_q) @ user_q.T).data
        user_Beta = torch.div(user_Beta, n_user)
        user_Beta = torch.softmax(user_Beta, dim=0)
        user = torch.zeros([n_user, embedding_size]).cuda()
        for i in range(n_user):
            for j in range(n_user_metapath):
                user[i] = user[i] + user_Beta[j] * user_H[j, i]
        # 产品
        product_metapath_guided_neighbors = metapath_guided_neighbors[1]
        n_product_metapath = len(product_metapath_guided_neighbors)
        # 产品可训练参数
        product_A = torch.full([n_product_metapath, n_product, n_user], -1e-9).cuda()
        product_H = torch.zeros([n_product_metapath, n_product, embedding_size]).cuda()
        product_Beta = torch.zeros(n_product_metapath).cuda()
        product_V, product_X, product_W_p = V[1], X[1], W_p[1]
        product_B_p, product_W_q, product_B_q = B_p[1], W_q[1], B_q[1]
        product_Q = Q[1]
        for i in range(n_product_metapath):
            product_v_p, product_x_p = product_V[i], product_X[i]
            product_w_p, product_b_p = product_W_p[i], product_B_p[i]
            for j in range(n_product):
                for k in product_metapath_guided_neighbors[i][j]:
                    product_A[i, j, k] = torch.tanh(
                        (product[j] @ product_v_p + user[k] @ product_w_p + product_b_p)) @ product_x_p.T
            product_A = torch.softmax(product_A, dim=2)
            for j in range(n_product):
                product_H[i, j] = product_H[i, j] + product[j]
                for k in product_metapath_guided_neighbors[i][j]:
                    product_H[i, j] = product_H[i, j] + product_A[i, j, k].data * user[k]
            product_w_q, product_b_q, product_q = product_W_q[i], product_B_q[i], product_Q[i]
            for j in range(n_product):
                product_Beta[i] = product_Beta[i] + (
                            torch.tanh(product_H[i, j] @ product_w_q + product_b_q) @ product_q.T).data
        product_Beta = torch.div(product_Beta, n_product)
        product_Beta = torch.softmax(product_Beta, dim=0)
        product = torch.zeros([n_product, embedding_size]).cuda()
        for i in range(n_product):
            for j in range(n_product_metapath):
                product[i] = product[i] + product_Beta[j] * product_H[j, i]
        return user, product


# 嵌入融合层
class EmbeddingFusionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softamx = nn.Softmax(dim=1)

    def forward(self, view_items, view_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        user_view_weight, product_view_weight = self.softamx(view_weight[0]), self.softamx(view_weight[1])
        # 视图级权重
        view_users, view_products = view_items[0], view_items[1]
        n_view, n_user, embedding_size = view_users.shape
        users = torch.zeros([n_user, embedding_size]).cuda()
        # 取得用户全局嵌入
        for i in range(n_user):
            for j in range(n_view):
                users[i] = users[i] + view_users[j, i] * user_view_weight[0, j]
        n_view, n_product, embedding_size = view_products.shape
        # 取得产品全局嵌入
        products = torch.zeros([n_product, embedding_size]).cuda()
        for i in range(n_product):
            for j in range(n_view):
                products[i] = products[i] + view_products[j, i] * product_view_weight[0, j]
        return users, products