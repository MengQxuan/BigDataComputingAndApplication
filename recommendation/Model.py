import numpy as np
from math import sqrt
from tqdm import tqdm
from matplotlib import pyplot as plt

class SVDModel:

    '''
        trainset: 训练集
        features: 特征维度
        epochs: 迭代次数
        lr: 学习率
        reguliaarization: 正则项
        UserBs: 用户偏置
        ItemBias: 商品偏置
        UserCharacteristicMatrix: 用户特征矩阵
        ItemCharacteristicMatrix: 商品特征矩阵
    '''

    def __init__(self, features=20, epochs=20, lr=0.002, regularization=0.025):
        self.features = features
        self.epochs = epochs
        self.lr = lr
        self.regularization = regularization
        self.UserBias = None
        self.ItemBias = None
        self.UserCharacteristicMatrix = None
        self.ItemCharacteristicMatrix = None

    def SVDtrain(self, trainset):
        self.trainset = trainset
        # 初始化
        rand = np.random.RandomState(0)
        self.UserBias = np.zeros(trainset.num_users, np.double)
        self.ItemBias = np.zeros(trainset.num_items, np.double)
        self.UserCharacteristicMatrix = rand.normal(0.0, 0.05, (trainset.num_users, self.features))
        self.ItemCharacteristicMatrix = rand.normal(0.0, 0.05, (trainset.num_items, self.features))
        mean_rating = trainset.calculate_mean_rating()
        regularization = self.regularization
        x = []
        y = []
        for epoch in range(self.epochs):
            print(f"第 {epoch + 1} 轮迭代...")
            # 更新学习率
            lr = self.lr / (2.0 ** epoch)
            sum_eui_squared = 0.0
            num_ratings = 0
            # 随机梯度下降
            for uid, iid, rating in tqdm(trainset.get_all_ratings()):
                num_ratings += 1
                # 求内积
                multi = np.dot(self.ItemCharacteristicMatrix[iid], self.UserCharacteristicMatrix[uid])
                # 预测
                predict = mean_rating + self.UserBias[uid] + self.ItemBias[iid] + multi
                predict = min(max(predict, 0), 100)
                eui = rating - predict
                sum_eui_squared += eui ** 2
                # 更新
                self.UserBias[uid] += lr * (eui - regularization * self.UserBias[uid])
                self.ItemBias[iid] += lr * (eui - regularization * self.ItemBias[iid])
                for i in range(self.features):
                    temp = self.UserCharacteristicMatrix[uid, i]
                    self.UserCharacteristicMatrix[uid, i] += lr * (eui * self.ItemCharacteristicMatrix[iid, i] - regularization * temp)
                    self.ItemCharacteristicMatrix[iid, i] += lr * (eui * temp - regularization * self.ItemCharacteristicMatrix[iid, i])
            x.append(epoch + 1)
            y.append(sqrt(sum_eui_squared / num_ratings))
            print()
            print(f"train_RMSE: {sqrt(sum_eui_squared / num_ratings)}")
        plt.title("train_RMSE")
        plt.xlabel("epoch")
        plt.ylabel("RMSE")
        plt.plot(x, y)
        plt.savefig('train_RMSE.png')
        plt.show()

    def SVDpredict(self, testset):
        pred = []
        for user, item in tqdm(testset):
            score = self.trainset.calculate_mean_rating()
            if self.trainset.exist_user(user) and self.trainset.exist_item(item):
                uid = self.trainset.get_map_userid(user)
                iid = self.trainset.get_map_itemid(item)
                score += self.UserBias[uid] + self.ItemBias[iid] + np.dot(self.ItemCharacteristicMatrix[iid], self.UserCharacteristicMatrix[uid])
            elif self.trainset.exist_user(user):
                uid = self.trainset.get_map_userid(user)
                score += self.UserBias[uid]
            elif self.trainset.exist_item(item):
                iid = self.trainset.get_map_itemid(item)
                score += self.ItemBias[iid]
            score = min(max(score, 0), 100)
            pred.append(score)
        return pred
