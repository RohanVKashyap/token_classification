import torch
import sys
from os import path, makedirs
import pathlib
import json
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import Sequential, Linear, Module
#from sentence_transformers import SentenceTransformer
from typing import List, Dict, Callable
from custom_dataset import get_cooking_dataset, get_ucr_dataset, embed_bulk
from tqdm import tqdm
from collections import deque
torch.multiprocessing.set_start_method('spawn', force=True)
from toolz.curried import pipe, pluck, merge_with, identity, get, do
import time
from cached_embedder import embed
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print('device:{}'.format(DEVICE))

class ClassifierModel(torch.nn.Module):

    def __init__(self, model_name, embedding_dim, num_labels = 1, num_classes=1,
                 label2id =None, model_path = "./",  loss_fn = None, **kwargs):
        super(ClassifierModel, self).__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.embedding_dim  = embedding_dim
        self.num_labels  = num_labels
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.lr = deque(maxlen=25)
        self.starting_lr = None
        self.label2id = label2id

        model_type = "multi label" if self.num_labels > 1 else "" + " multi class" if self.num_classes > 1 else ""
        #print(f"Init {model_type}")
        self.model = Sequential(
            Linear(self.embedding_dim, 64),
            #Linear(64, 64),
            Linear(64, self.num_classes)
            #Linear(self.embedding_dim, 128),
            #Linear(128, self.num_classes)
        )
        #self.model.to("cuda")

    def forward(self, input):
        #print(input)
        return self.model(input['input_embeddings']).to(DEVICE)

    def fit(self, loader, loss_fn, optimizer, epochs):
        self.train()
        prev_loss = 0
        lr_change_attempt = 0
        for e in tqdm(range(1, epochs+1)):
            e_loss = 0
            e_loss_cum = 0
            for input in loader:
                optimizer.zero_grad()
                predicted = self(input)
                #print(predicted.tolist())
                #print(input['label'].tolist())
                loss = loss_fn(predicted, input['label'].long().squeeze())
                loss.backward()
                optimizer.step()
                #temp_loss = loss.item()
                e_loss_cum += loss
                #print(f"temp_loss : {temp_loss} epoch_loss : {e_loss} lr : {optimizer.defaults['lr']}")
            e_loss = e_loss_cum.item()
            loss_change = round(( (prev_loss - e_loss)/(prev_loss or e_loss)) * 100, 4)
            print(f"epoch_loss : {round(e_loss, 4)} loss_change : {loss_change}%  lr : {optimizer.defaults['lr']}")
            cur_lr = optimizer.defaults['lr']

            self.starting_lr = self.starting_lr or cur_lr
            self.lr.append(cur_lr)
            if prev_loss > 0 and e_loss > prev_loss:
                if lr_change_attempt > 1 :
                    optimizer.defaults['lr'] = round(cur_lr/ 1.1, 4)
                    lr_change_attempt = 0
                else:
                    lr_change_attempt += 1
            elif len(self.lr) > 10 and loss_change > 1 and all([ cur_lr == i for i in self.lr]):
                #optimizer.defaults['lr'] = min(round(cur_lr * 1.1, 4), self.starting_lr)
                optimizer.defaults['lr'] = min(round(cur_lr * 1.1, 4), max(self.lr))
            #print(self.lr, cur_lr)
            prev_loss = e_loss


    @classmethod
    def from_pretrained(cls, model_path):
        model_config = json.load(open(path.join(model_path, 'model_config.json')))
        state_dict_filename = path.join(model_path, model_config['state_dict'])
        model = cls(**model_config, model_path = model_path)
        print(f"Loading file from {state_dict_filename}")
        d = torch.load(state_dict_filename)
        model.load_state_dict(d)
        return model

    def save(self, model_path ="./"):
        full_path = path.join(model_path, self.model_name, 'model_bin.pt')
        model_bin_filename = "model_bin.pt"
        def ensure_dir(file_path):
            directory = path.dirname(file_path)
            if not path.exists(directory):
                makedirs(directory)

        ensure_dir(full_path)
        print(f"Save model {self.model_name} {full_path}")
        model_config = { 'model_name' : self.model_name,
                         'embedding_dim' : self.embedding_dim,
                         'num_labels' : self.num_labels,
                         'num_classes' : self.num_classes,
                         'state_dict' : 'model_bin.pt',
                         'label2id' : self.label2id
                        }
        with open(path.join(path.dirname(full_path), 'model_config.json'), 'w' ) as jf:
            json.dump( model_config, jf, indent=6)

        torch.save(self.state_dict(), path.join(model_path, self.model_name, model_bin_filename ) )
        #torch.save(self, full_path)

    def finetune(self, ds : torch.utils.data.Dataset):
        self.train_model( ds , self.num_labels, self.num_classes, self.label2id,
                        model_name = self.model_path,  model_path=self.model_path, finetune = True)
    @classmethod
    def train_model(cls, ds : torch.utils.data.Dataset, num_labels, num_classes, label2id,
                    model_name = None,  model_path="./", finetune = False):
        loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)
        train_dataset = ds
        train_loader = DataLoader(dataset=train_dataset,
                                #num_workers=4,
                                #shuffle=True,
                                batch_size=200000,
                                collate_fn = collate_fn
                                #sampler=weighted_sampler
                                    )
        LEARNING_RATE = 0.15
        classifier_model = None
        if finetune:
            print("Partial  model training ", model_name, model_path)
            classifier_model = cls.from_pretrained(model_path)
        else:
            print("New model training", model_name, model_path)
            classifier_model = cls(model_name, 384,  model_path=model_path,
                                    num_labels=num_labels, num_classes=num_classes,
                                   label2id=label2id )
        #classifier_model = classifier_model.to("cuda")
        classifier_model.to(DEVICE)
        optimizer = optim.Adam(classifier_model.parameters(), lr=LEARNING_RATE)

        classifier_model.fit(train_loader, loss_fn, optimizer, 2)
        classifier_model.save()

def validate_accuracy():
    pass


def collate_fn(data):
    t1 = time.time()
    ret = merge_with(identity, *data)
    new_input=[]
    for j in ret['input_text']:
        new_input.append(embed(tuple(j.split())))

    #ret['input_embeddings']=torch.stack(new_input).view(-1,384)
    ret['input_embeddings']=torch.cat(new_input).view(-1,384)
    #ret['label'] = torch.stack(ret['label']).squeeze(dim=1).view(-1,1).to(DEVICE)
    ret['label'] = torch.cat(ret['label'],dim=1).squeeze(dim=1).view(-1,1).to(DEVICE)
    print(ret['input_embeddings'].shape, ret['label'].shape)
    t2 = time.time()
    return ret


def test_train_ucr_new():
    model_name = "ucr_classifier_model"
    classifier_model = ClassifierModel.train_model(*get_ucr_dataset(),model_name = model_name,
                                                   finetune = False)
    classifier_model = ClassifierModel.from_pretrained(model_path = "./" + model_name)

def test_train_ucr_retrain():
    model_name = "ucr_classifier_model"
    classifier_model = ClassifierModel.from_pretrained(model_path = "./" + model_name)
    ds, num_labels, num_classes, label2id, = get_ucr_dataset()
    classifier_model.finetune(ds)

def test_prediction():
    st_model_path = 'models/sentence_transformers/all-MiniLM-L6-v2'
    st = SentenceTransformer(st_model_path , device=DEVICE)
    model_name = "ucr_classifier_model"
    classifier_model = ClassifierModel.from_pretrained(model_path = "./" + model_name).to(DEVICE)
    id2label = { int(v): k for k, v in classifier_model.label2id.items()}
    embed  = lambda x : st.encode(x, convert_to_tensor=True).to(DEVICE)
    sm = torch.nn.Softmax()

    def predict(text):
        make_input  = lambda x : {'input_embeddings' : embed(x)}
        return pipe(text , make_input, classifier_model,
                    #do(lambda x: print(sm(x ).topk(3))),
                    #do(lambda x: print(torch.nn.functional.normalize(x, dim=0))),
                    #do(lambda x: print(torch.sigmoid(x, ))),
                    #do(lambda x: print(torch.logit(x, ))),
                    torch.argmax, int, id2label.get )
    #examples =[
        #'where is the nearest atm',
        #'how to open a loan account',
        #'hello how are you ',
        #'how to change the PIN number on my card']

    #labels=['qry-atmbranchlocator','faq','invalid','txn-pinsettings']

    test_data = pd.read_csv('ucr_test_data5.tsv',delimiter = '\t',
                           converters = {'label': pd.eval})
    
    test_sentences = test_data['query'].values.tolist()
    test_labels = test_data['label'].values.tolist()
    new_labels = [item for sublist in test_labels for item in sublist]

    res = [(i,predict(i)) for t in test_sentences for i in t.split()] 

    total_correct=0.0    
    for i, (j,k) in zip(new_labels,res):
        if str(i) == str(k):
            total_correct+=1

    accuracy=(total_correct/len(new_labels))*100
    print('accuracy:',accuracy)             

if __name__=='__main__':
   if sys.argv[1]=='train':
        print('Training in Progress')
        test_train_ucr_new()
   else:
       print('Testing in Progress')
       test_prediction() 


# def test_train_cooking():
#     model_name = "cooking_classifier_model"
#     classifier_model = ClassifierModel.train_model(*get_cooking_dataset(),model_name)
#     classifier_model = ClassifierModel.from_pretrained(model_name)
