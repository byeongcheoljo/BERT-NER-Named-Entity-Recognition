import pandas as pd
import numpy as np
from tqdm import tqdm, trange

data = pd.read_csv("nerDataSet/ner_dataset.csv", encoding="latin1").fillna(method="ffill")
print(data.head(10))


class SentenceGetter():
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        self.agg_func = lambda s: [(w,p,t) for w,p,t in zip(s['Word'].values.tolist(), s['POS'].values.tolist(), s['Tag'].values.tolist())]

        self.grouped = self.data.groupby("Sentence #").apply(self.agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped("sentece:{}".format(self.n_sent))
            self.nsent +=1
            return s
        except:
            return None


getter = SentenceGetter(data)
print(getter)
print(getter.sentences)
