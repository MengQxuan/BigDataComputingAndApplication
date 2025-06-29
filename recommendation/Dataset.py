class TrainSet:
    def __init__(self, data):
        # data: list of (user, item, rating) tuples
        self.raw_data = data
        self.user2id = {}
        self.item2id = {}
        self.id2user = {}
        self.id2item = {}
        self.user_ratings = {}
        self.item_ratings = {}

        user_cnt = 0
        item_cnt = 0
        for user, item, rating in data:
            if user not in self.user2id:
                self.user2id[user] = user_cnt
                self.id2user[user_cnt] = user
                user_cnt += 1
            if item not in self.item2id:
                self.item2id[item] = item_cnt
                self.id2item[item_cnt] = item
                item_cnt += 1
            uid = self.user2id[user]
            iid = self.item2id[item]
            self.user_ratings.setdefault(uid, []).append((iid, rating))
            self.item_ratings.setdefault(iid, []).append((uid, rating))

        self.num_users = user_cnt
        self.num_items = item_cnt

    def get_all_ratings(self):
        for user, item, rating in self.raw_data:
            uid = self.user2id[user]
            iid = self.item2id[item]
            yield uid, iid, rating

    def calculate_mean_rating(self):
        total = sum(r for _, _, r in self.raw_data)
        return total / len(self.raw_data)

    def exist_user(self, user: str) -> bool:
        return user in self.user2id

    def exist_item(self, item: str) -> bool:
        return item in self.item2id

    def get_map_userid(self, user: str) -> int:
        return self.user2id[user]

    def get_map_itemid(self, item: str) -> int:
        return self.item2id[item]
