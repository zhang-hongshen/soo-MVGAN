import pymongo
import torch
from flask import Flask, jsonify, request
from training.model import Predictor
import config

app = Flask(__name__)

predict_num = 7

@app.route('/predict', methods=['post', 'get'])
def predict():
    # 获取传入的userId
    if request.method == 'post':
        user_id = request.form.get("user_id")
    else:
        user_id = request.args.get("user_id")
    # 创建mongodb客户端
    mongodb_client = pymongo.MongoClient(host=config.MONGODB_HOST, port=config.MONGODB_PORT)
    mongodb_client['admin'].authenticate(config.MONGODB_USERNAME, config.MONGODB_PASSWORD)
    mydb = mongodb_client['soo']
    t_user = mydb["soo_user"]
    t_product = mydb["soo_product"]
    # 获取用户embedding
    user = t_user.find_one({'user_id': user_id})
    if user is None or user.get('embedding') is None:
        user_embed = torch.nn.Embedding(1, 64)
        user_embedding = user_embed([0]).cuda()
        t_user.insert_one({'user_id': user_id, 'embedding': user_embedding})
    user_embedding = user.get('embedding')
    # python数据结构与pytorch数据结构转化
    user_embedding_tensor = torch.unsqueeze(torch.tensor(user_embedding).cuda(), dim=0)
    products = t_product.find()
    product_id_list = []
    product_embedding_list = []
    for product in products:
        product_id_list.append(product.get('product_id'))
        product_embedding_list.append(product.get('embedding'))
    product_embedding_tensor = torch.tensor(product_embedding_list).cuda()
    # 预测
    output = Predictor().predict(user_embedding_tensor, product_embedding_tensor)
    # 筛选得分最高的K个产品ID
    _, indices = torch.sort(output, descending=True, dim=1)
    topK_indices = indices.squeeze(0).cpu().numpy().tolist()[:predict_num]
    # 将结果序列化为json格式
    return jsonify([product_id_list[index] for index in topK_indices])


if __name__ == '__main__':
    app.run(debug=True, port=8383)
