import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

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
sentences = [[word[0] for word in sentence]for sentence in getter.sentences]
print(sentences[0])
labels = [[word[2] for word in sentence]for sentence in getter.sentences]
print(labels[0])

tag_values = list(set(data["Tag"].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}
print(tag2idx)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))

max_length = 100
batch_size = 32

tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)

        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


print(tokenized_texts[1])
print(labels[1])
