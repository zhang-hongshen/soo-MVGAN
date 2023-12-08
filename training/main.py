import pandas as pd
import pymongo
import redis
import torch
from torch.utils.data import DataLoader

import config
from dataset import MyDataset
from model import MultiViewGraphAttentionNetWork, Predictor

'''
redis客户端初始化
'''
redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True, db=0, password=config.REDIS_PASSWORD)
torch.autograd.set_detect_anomaly(True)
'''
读取数据
'''
user_dataset = MyDataset('../soo_user.csv', ['user_id'])
product_dataset = MyDataset('../soo_product.csv', ['product_id', 'departure', 'destination', 'price'])
interaction_dataset = MyDataset('../soo_interaction.csv', ['user_id', 'product_id'])
user_search_dataset = MyDataset('../soo_user_search.csv', ['user_id', 'departure', 'destination'])
product_departure = MyDataset('../soo_product_departure.csv', ['product_id', 'departure', ])
interaction_df = interaction_dataset.data
product_df = product_dataset.data
user_df = user_dataset.data
user_search_df = user_search_dataset.data
product_departure_df = product_departure.data
city_list = list(redis_client.smembers('city'))
user_id_list = user_df['user_id'].unique().tolist()
product_id_list = product_df['product_id'].unique().tolist()
n_user = len(user_id_list)
n_product = len(product_id_list)
n_city = len(city_list)
# 价格分成11个区间
n_price = 11
'''
将实际的信息转换为数字
'''
interaction_df['user_id'] = interaction_df['user_id'].apply(lambda x: user_id_list.index(x))
interaction_df['product_id'] = interaction_df['product_id'].apply(lambda x: product_id_list.index(x))
user_search_df['user_id'] = user_search_df['user_id'].apply(lambda x: user_id_list.index(x))
user_search_df['departure'] = user_search_df['departure'].apply(lambda x: city_list.index(x))
user_search_df['destination'] = user_search_df['destination'].apply(lambda x: city_list.index(x))
user_df['user_id'] = user_df['user_id'].apply(lambda x: user_id_list.index(x))
product_df['product_id'] = product_df['product_id'].apply(lambda x: product_id_list.index(x))
product_df['departure'] = product_df['departure'].apply(lambda x: city_list.index(x))
product_df['destination'] = product_df['destination'].apply(lambda x: city_list.index(x))
product_departure_df['product_id'] = product_departure_df['product_id'].apply(lambda x: product_id_list.index(x))
product_departure_df['departure'] = product_departure_df['departure'].apply(lambda x: city_list.index(x))
bins = [i * 2000 for i in range(n_price)]
bins.append(max(product_df['price']))
product_df['price'] = pd.cut(product_df['price'], bins, labels=[i for i in range(n_price)])
user_product = torch.zeros(n_user, n_product).cuda()
user_departure = torch.zeros(n_user, n_city).cuda()
user_destination = torch.zeros(n_user, n_city).cuda()
product_departure = torch.zeros(n_product, n_city).cuda()
product_destination = torch.zeros(n_product, n_city).cuda()
product_price = torch.zeros(n_product, n_price).cuda()


def to_matrix(df, matrix):
    for i, j in df.drop_duplicates().to_records(
            index=False).tolist():
        matrix[i][j] = 1


'''
转换成邻接矩阵
'''
to_matrix(interaction_df[['user_id', 'product_id']], user_product)
to_matrix(user_search_df[['user_id', 'departure']], user_departure)
to_matrix(user_search_df[['user_id', 'destination']], user_destination)
to_matrix(product_departure_df[['product_id', 'departure']], product_departure)
to_matrix(product_df[['product_id', 'destination']], product_destination)
to_matrix(product_df[['product_id', 'price']], product_price)
# 预处理结束赋值
interaction_dataset.data = interaction_df.sample(frac=1).reset_index(drop=True)
product_dataset.data = product_df
user_dataset.data = user_df
user_search_dataset.data = user_search_df
# product_departure.data = product_departure_df
'''
初始化ID embedding
'''
embedding_size = 64
user_tensor = torch.tensor(user_df['user_id'].drop_duplicates().tolist())
user_embed = torch.nn.Embedding(n_user, embedding_size)
user_embedding = user_embed(user_tensor).cuda()
product_tensor = torch.tensor(product_df['product_id'].drop_duplicates().tolist())
product_embed = torch.nn.Embedding(n_product, embedding_size)
product_embedding = product_embed(product_tensor).cuda()
'''
参数合并便于传入
'''
user_attr = [user_departure, user_destination]
product_attr = [product_departure, product_destination, product_price]
'''
训练
'''
# 模型定义
model = MultiViewGraphAttentionNetWork(embedding_size).cuda()
batch_size = 256
# 训练集占比
train_frac = 0.7
# 训练集大小
train_size = int(interaction_dataset.__len__() * train_frac)
# 测试集大小
test_size = interaction_dataset.__len__() - train_size
train_interaction_dataset, test_interaction_dataset = torch.utils.data.random_split(interaction_dataset,
                                                                                    [train_size, test_size])
train_dataloader = DataLoader(train_interaction_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_interaction_dataset, batch_size=batch_size, shuffle=True)

lr = 0.0001  # 学习速率
n_epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()  # 二分类交叉熵
model.train()

'''
mongodb连接初始化
'''
mongodb_client = pymongo.MongoClient(host=f"mongodb://{config.MONGODB_USERNAME}:{config.MONGODB_PASSWORD}@{config.MONGODB_HOST}:{config.MONGODB_PORT}/", port=config.MONGODB_PORT)
mydb = mongodb_client["soo"]
t_user = mydb["soo_user"]
t_product = mydb["soo_product"]

for i in range(n_epochs):
    step = 0
    for data in train_dataloader:
        train_user_product = torch.zeros(n_user, n_product).cuda()  # 保存训练后的u-v矩阵
        # 将数据转换为邻接矩阵
        n = len(data[0])
        for j in range(n):
            train_user_product[data[0][j]][data[1][j]] = 1
        optimizer.zero_grad()
        users, products = model([user_embedding, product_embedding],
                                train_user_product, user_attr, product_attr)
        print("结果保存至mongodb")
        for j in range(n_user):
            t_user.replace_one({'user_id': user_id_list[j]},
                               {'user_id': user_id_list[j], 'embedding': users[j].tolist()},
                               upsert=True)
        for j in range(n_product):
            t_product.replace_one({'product_id': product_id_list[j]},
                                  {'product_id': product_id_list[j], 'embedding': products[j].tolist()},
                                  upsert=True)
        output = Predictor().predict(users, products)
        loss = criterion(output, train_user_product).cuda()
        loss.backward(retain_graph=True)
        optimizer.step()
        print(f'Epoch[{i}/{n_epochs}],Step[{step}],Loss={loss.data}')
        step = step + 1
writer.add_scalar(‘loss/train_loss’, losses.val, global_step=n_epochs)
print('训练完成')
torch.save(model, 'mvgan.pth')
