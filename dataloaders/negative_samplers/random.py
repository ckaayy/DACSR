from .base import AbstractNegativeSampler

from tqdm import trange

import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        user_list = list(self.train_target.keys())
        #print(user_list)

        for k in trange(len(user_list)):
            user = user_list[k]
        #     if isinstance(self.train_target[user][1], tuple):
        #             seen = set(x[0] for x in self.train_target[user])
        #             seen.update(x[0] for x in self.val[user])
        #             seen.update(x[0] for x in self.test[user])
        #     else:
            seen = set(self.train_target[user])
            if self.val.get(user) is not None:
                seen.update(self.val[user])
            if self.test.get(user) is not None:
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count)+1 
                while item in seen or item in samples:
                        item = np.random.choice(self.item_count)+1 
                samples.append(item)

                negative_samples[user] = samples
                
        return negative_samples

