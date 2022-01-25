
from itertools import takewhile
from toolz.curried import ( pipe, curry, map, compose_left, valmap,
                            compose, concat, unique, interpose, get, do, merge,
                            frequencies, memoize)
import toolz
from operator import methodcaller
from datasets import Dataset
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import time

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
t1  = time.time()
st_model_path = 'models/sentence_transformers/all-MiniLM-L6-v2'
st = SentenceTransformer(st_model_path , device=DEVICE)
st.eval()
t2  = time.time()
#print(f"model loading  dataset {t2-t1}")

@dataclass
class ClassifierDataset(torch.utils.data.Dataset):
    df : pd.DataFrame
    text_col : str
    cache : Dict = None

    def __post_init__(self):
        #texts = self.df[self.text_col].tolist()
        #print("post_init")
        self.cache = {}
        t1 = time.time()
        values = self.df.values.tolist()

        for idx in range(len(values)):
            #l = self.df.iloc[idx]
            ret = {'input_text' : values[idx][0], 'label' : torch.tensor(values[idx][1:]).to(DEVICE) }
            #ret = {'input_text' : l[self.text_col], 'label' : torch.tensor(l[1:]) }
            self.cache[idx] = ret

        t2 = time.time()
        #print(f"Loading data {t2-t1}")

    def __getitem__(self, idx):
        #print(f"Dataset item {idx}")
        #return {'input_text' : l[self.text_col], 'label' : torch.tensor(l[1:]) }
        cached_data = self.cache.get(idx, None)
        if cached_data :
            return cached_data
        else:
            l = self.df.iloc[idx]
            #embeddings = st.encode(l[self.text_col], convert_to_tensor=True, device=DEVICE)
            ret = {'input_text' : l[self.text_col], 'label' : torch.tensor(l[1:]).to(DEVICE) }
            # ret = {'input_text' : l[self.text_col], 'label' : torch.tensor(l[1:]).to(DEVICE),
            #         'input_embeddings' : embeddings}
            self.cache[idx] = ret
            return ret

    def __len__(self):
        return len(self.df) - 1

def get_cooking_data():
    partitionby = curry(toolz.recipes.partitionby)
    replace = curry(methodcaller, 'replace')
    make_labels = compose( str.split, replace("__label__", ''))
    make_pairs = compose_left(str.split,
                    partitionby(lambda x: x.startswith("__label") ),
                    map(" ".join),  tuple, reversed, tuple)

    data = pipe(open('cooking_data.txt'),
                map(make_pairs),
                dict,
                valmap(make_labels))
    return data

def get_cooking_labels():
    return pipe(get_cooking_data(), dict.values, concat,
                  unique, sorted, list )

def write_cooking_labels():
    pipe(get_cooking_labels(), interpose("\n"), list,
                  open('cooking_labels.txt', 'w').writelines)

def write_cooking_labels_freq():
    pipe(get_cooking_data(), dict.values, concat, frequencies, dict.items,
         curry(sorted, key=get(1), reverse=True), dict,
         valmap(str), dict.items,
         map(" ".join), interpose("\n"),
                  open('cooking_labels_frequencies.txt', 'w').writelines)

def write_cooking_labels_freq():
    pipe(get_cooking_data(), dict.values, concat, frequencies, dict.items,
         curry(sorted, key=get(1), reverse=True), map(lambda x: x[0] +" "+ str(x[1])),
         interpose("\n"),
                  open('cooking_labels_frequencies.txt', 'w').writelines)

def embed(item : Dict):
    i = item['input_text']
    item['input_embeddings']  = st.encode(i, convert_to_tensor=True)
    return item

@memoize
def embed_bulk(texts):
    return st.encode(texts, convert_to_tensor=True, device=DEVICE)

#write_cooking_labels_freq()

def get_dataframe():
    data = get_cooking_data()
    labels = get_cooking_labels()
    all_labels = ['text', *labels]
    d = [merge({'text' : t}, {l:1 for l in lbs} ) for t, lbs in data.items()]
    df = pd.DataFrame(d, columns=all_labels)
    df = df.fillna(0)
    #return Dataset.from_pandas(df)
    print(df)
    return df

def get_cooking_dataset():
    df = get_dataframe()
    num_labels = len(df.columns) - 1
    all_classes = { v : float(i) for i, v in enumerate(df.columns) }
    num_classes = 1
    ds = ClassifierDataset(df, 'text')
    #ds = ds.map(embed)
    return ds, num_labels, num_classes, all_classes

def get_ucr_dataset():
    #df = pd.read_csv('ucr_data.tsv', delimiter='\t')
    #df = pd.read_csv('ucr_train_data2.tsv',delimiter='\t')
    df = pd.read_csv('ucr_test_data5.tsv', delimiter = '\t', converters = {'label': pd.eval})
    num_labels = 1
    all_labels = [item for sublist in df['label'].values.tolist() for item in sublist]
    all_classes = { v : float(i) for i, v in enumerate(set(all_labels))}
    df['label'] = [[all_classes[j] for j in i] for i in df['label'].values.tolist()]
    #df['label'] = [all_classes[v] for v in df.label.tolist()]
    num_classes = len(all_classes)
    ds = ClassifierDataset(df, 'query')
    #ds = ds.map(embed)
    return ds, num_labels, num_classes, all_classes

#ds = get_cooking_dataset()
#print(next(iter(ds)))
