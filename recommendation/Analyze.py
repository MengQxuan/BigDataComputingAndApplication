def parse_train_file(file_path):
    """
    解析训练文件，统计用户集、物品集、评分总数、评分均值、最大评分、最小评分。
    """
    user_set = set()
    item_set = set()
    score_num = 0
    score_sum = 0.0
    max_score = float('-inf')
    min_score = float('inf')

    with open(file_path, 'r') as f:
        lines = f.readlines()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1

        if '|' not in line:
            continue

        userid, num_ratings = line.split('|')
        num_ratings = int(num_ratings)
        user_set.add(userid)
        score_num += num_ratings

        for _ in range(num_ratings):
            line = lines[index].strip()
            index += 1
            itemid, score = line.split(' ')
            score = float(score)

            item_set.add(itemid)
            score_sum += score
            max_score = max(max_score, score)
            min_score = min(min_score, score)

    return {
        'user_count': len(user_set),
        'item_count': len(item_set),
        'score_num': score_num,
        'score_sum': score_sum,
        'max_score': max_score,
        'min_score': min_score
    }

def AnalyzeData():
    stats = parse_train_file('train.txt')
    print('用户数量:', stats['user_count'])
    print('商品数量:', stats['item_count'])
    print('评分数量:', stats['score_num'])
    print('最高评分:', stats['max_score'])
    print('最低评分:', stats['min_score'])
    print('评分均值:', stats['score_sum'] / stats['score_num'])
