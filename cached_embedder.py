import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Callable
from dataclasses import dataclass
from toolz.curried import *
import fcntl
import torch
import time
from functools import wraps

#DEVICE = 'cuda'
#DEVICE = 'cpu'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'all-MiniLM-L6-v2'
st_model_path = 'models/sentence_transformers/all-MiniLM-L6-v2'
t1  = time.time()
st = SentenceTransformer(st_model_path, device=DEVICE)
st.eval()
t2  = time.time()
#print(f"model loading {t2-t1}")

CACHE_NP_FILE_NAME = f"{model_name}_cache.npy"
CACHE_INDEX_FILE_NAME = f"{model_name}_cache.txt"


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        #print(f'func:{f.__name__} took: {te-ts} sec'  )
        return result
    return wrap

@dataclass
class Texts:
    texts : List[str]
    cached_indices_map : dict = None
    noncached_indices_map : dict = None


#@memoize
def embed_bulk(texts):
    return st.encode(texts, convert_to_numpy=True, device=DEVICE) if len(texts) > 0 else []


@timing
def search_text_in_cache(sentences):
    try:
        with open(CACHE_INDEX_FILE_NAME) as f:
            lines = pipe(f.readlines(), map(str.strip), list)
            #d = {s:i for i, s in enumerate(lines)}
            d = dict( zip(lines, range(0, len(lines))) )
            index = lambda x: d.get(x, -1)
            #index = excepts(ValueError, lambda x: lines.index(x), lambda _: -1)
            return pipe(sentences, map(str.strip),  map(index), list )
    except Exception:
        return [-1 for i in sentences]

@timing
def split_cached_noncached(idx):
    #print("split cached and not cachec idx ", idx)
    return {i:ci for i, ci in enumerate(idx) if ci != -1}, {i:ci for i, ci in enumerate(idx) if ci == -1}


@memoize
def embed(texts):
    t1 = time.time()
    cached, noncached = pipe(texts, search_text_in_cache, split_cached_noncached)
    embs = pipe(get(list(noncached.keys()), texts), list, embed_bulk)
    embs_dict = { idx: emb for idx, emb in zip(noncached, embs)}
    ret_dict = merge(get_from_cache(cached), embs_dict)
    #print(" combined keys for cached and non cached", ret_dict.keys())
    embeddings = [ ret_dict[i] for i in range(len(texts))]
    if len(noncached) > 0:
        noncached_texts = pipe(texts, get(list(noncached.keys())), list)
        write_to_cache(noncached_texts, embs)
    t2 = time.time()
    ret = torch.tensor(np.array(embeddings)).to(DEVICE)
    t3 = time.time()
    #print(f"Fetching from cache tensors {t2-t1}" )
    #print(f"Making tensors {t3-t2}" )
    return ret

@timing
def get_from_cache(cached):
    try:
        n = np.load(CACHE_NP_FILE_NAME)
        v = list(cached.values())
        k = list(cached.keys())
        return { a:b for a, b in zip( k, n[v] ) }
        #return {i:n[v] for i, v in cached.items()}
    except Exception as e:
        print("Exception in get_from cached", e)
        return {}

@timing
def write_to_cache(sentences, embeddings):
    #print(f"Write len of sentence {len(sentences)} and len embeddings {len(embeddings)}")
    n = None
    try:
        n = np.load(CACHE_NP_FILE_NAME)
    except Exception:
        print("Error in opening NP cache file")

    with open(CACHE_INDEX_FILE_NAME, "a+") as g:
        fcntl.flock(g, fcntl.LOCK_EX)
        try:
            #print(sentences)
            pipe(sentences, map(lambda x: x+'\n'), list, g.writelines)
            #print(len(embeddings))
            #print(len(np.array(embeddings)))
            new_n = np.append(n, embeddings, axis=0) if n is not None else np.array(embeddings)
            #print("new_n shape", new_n.shape)
            np.save(CACHE_NP_FILE_NAME, new_n)

        finally :
            fcntl.flock(g, fcntl.LOCK_UN)
