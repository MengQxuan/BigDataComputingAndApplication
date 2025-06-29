from math import sqrt
from Dataset import TrainSet
from Model import SVDModel
import random

# 计算RMSE
def calculateRMSE(pred, oracle):
    sum_sq = 0.0
    for i in range(len(pred)):
        sum_sq += (oracle[i] - pred[i])**2
    return sqrt(sum_sq / len(pred))

# 加载train.txt
def load_train_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    user = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '|' in line:
            user = line.split('|')[0]
        else:
            item, score = line.split()
            data.append((user, item, int(score)))
    return data

# 加载test.txt
def load_test_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    user = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '|' in line:
            user = line.split('|')[0]
        else:
            item = line.strip()
            data.append((user, item))
    return data

# 划分训练集/验证集
def split_train_and_val(dataset, train_ratio, val_ratio):
    assert train_ratio + val_ratio == 100
    random.seed(1)
    trainset = []
    valset = []
    for i in range(len(dataset)):
        rand = random.randint(0, 100)
        if rand == val_ratio:
            valset.append(dataset[i])
        else:
            trainset.append(dataset[i])
    return trainset, valset

def train():
    dataset = load_train_data('train.txt')
    print("开始划分训练集、验证集...")
    trainset_data, valset = split_train_and_val(dataset, 99, 1)
    print("划分完毕！")

    trainset = TrainSet(trainset_data)
    SVD = SVDModel()
    print("开始训练...")
    SVD.SVDtrain(trainset)
    print("训练完成！")

    # 验证集预测
    val = []
    oracle = []
    for (userid, itemid, score) in valset:
        val.append([userid, itemid])
        oracle.append(score)
    pred = SVD.SVDpredict(val)
    RMSE = calculateRMSE(pred, oracle)
    print('val_RMSE:', RMSE)

    # 测试集预测
    testset = load_test_data('test.txt')
    rating_counts = {}
    for user, _ in testset:
        rating_counts[user] = rating_counts.get(user, 0) + 1

    print("开始预测...")
    pred = SVD.SVDpredict(testset)
    print()
    print("预测完成！")

    # 保存结果
    print("保存结果...")
    with open('Result.txt', 'w+', encoding='utf-8') as f:
        now_user = ""
        for i in range(len(testset)):
            user, item = testset[i]
            score = pred[i]
            if user != now_user:
                now_user = user
                f.write(user + '|' + str(rating_counts[user]) + '\n')
                f.write(item + '  ' + str(score) + '\n')
            else:
                f.write(item + '  ' + str(score) + '\n')
    print("保存完毕！")

if __name__ == "__main__":
    train()
