import numpy as np
import pandas as pd
from string import punctuation
import re
import os
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator, Iterator
from torchtext.vocab import GloVe

import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
nltk.download('punkt')
nltk.download('wordnet')
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from basic_lstm import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print(device)
torch.cuda.empty_cache()

def get_embed_options(path_to_file):
  data = open(path_to_file, 'r')
  options = []
  lines = data.readlines()
  for line in lines:
    pieces = line.strip().split()
    pieces[1] = int(pieces[1])
    # print(line, pieces, tuple(pieces))
    options.append(tuple(pieces))
  return options

embed_options = get_embed_options("embedding_options.txt")
print(embed_options)

# set up image data
train_im = np.load("data_vectors/train-images-set.npy")
val_im = np.load("data_vectors/val-images-set.npy")
test_im = np.load("data_vectors/test-images-set.npy")

train_im = np.moveaxis(train_im, -1, 1)
val_im = np.moveaxis(val_im, -1, 1)
test_im = np.moveaxis(test_im, -1, 1)

# text prep

# embedding_glove = GloVe(name='6B', dim=100)
# print(embedding_glove['princess'])

train_text = pd.read_csv("data_vectors/train-text-set-clean-2.tsv",sep='\t')
val_text = pd.read_csv("data_vectors/val-text-set-clean-2.tsv",sep='\t')
test_text = pd.read_csv("data_vectors/test-text-set-clean-2.tsv",sep='\t')

train_text['label'] = train_text['label'].apply(lambda cat: int(cat))
val_text['label'] = val_text['label'].apply(lambda cat: int(cat))
test_text['label'] = test_text['label'].apply(lambda cat: int(cat))


### following code is for text data statistics
# print(len(train_text[train_text['label'] == 1]),len(train_text[train_text['label'] == 0]) )
# print(len(val_text[val_text['label'] == 1]),len(val_text[val_text['label'] == 0]) )
# print(len(test_text[test_text['label'] == 1]),len(test_text[test_text['label'] == 0]) )

# all_data = pd.read_csv("all_text_clean.txt", sep='\t')

# all_text = train_text.append(val_text)
# all_text = all_text.append(test_text)
# all_text.to_csv("all_text_clean.txt",sep='\t')
# pos_text = all_text[all_text['label'] == 1]
# neg_text = all_text[all_text['label'] == 0]
# pos_text.to_csv("pos_text_clean.txt",sep='\t')
# neg_text.to_csv("neg_text_clean.txt",sep='\t')

# train_text = train_text.loc[train_text['text'].notnull(), ['text','label']]
# val_text = val_text.loc[val_text['text'].notnull(), ['text','label']]
# test_text = test_text.loc[test_text['text'].notnull(), ['text','label']]

# test_text.to_csv('test-text-set-clean-2.tsv',sep='\t')
# val_text.to_csv('val-text-set-clean-2.tsv',sep='\t')
# train_text.to_csv('train-text-set-clean-2.tsv',sep='\t')

# print("lengths: ", len(train_text), len(val_text), len(test_text))

# text_field = Field(
#     sequential=True,
#     tokenize='basic_english', 
#     lower=True,
#     include_lengths=True,
#     batch_first=True,
#         )
# label_field = LabelField(dtype = torch.float,batch_first=True)

# fields = [(None, None),(None, None),('text',text_field),('label', label_field)]

# train, valid, test = TabularDataset(path="", train='all_data_clean-2.txt', validation='data_vectors/val-text-set-clean.tsv', test='data_vectors/test-text-set-clean.tsv', 
# format='TSV', fields=fields, skip_header=True)
# text_field.build_vocab(
#         train, 
#         vectors='glove.twitter.27B.200d'
#     )
# label_field.build_vocab(train)

# #No. of unique tokens in text
# print("Size of TEXT vocabulary:",len(text_field.vocab))

# #No. of unique tokens in label
# print("Size of LABEL vocabulary:",len(label_field.vocab))

# #Commonly used words
# print(text_field.vocab.freqs.most_common(15))  



## following to investiage where nulls are
# for index,row in train_text.iterrows():
#     print(type(row['text']),row['text'], row['label'])
#     if len(str(row['text'])) == 0:
#         print(row['text'], row['label'])
# print(train_text['text'].notnull())


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def experiment_word_emb():
  base_folder = "word_emb_experiments"
  cur_exp = "full_experiment_sgd_.1_2"
  exp_folder = os.path.join(base_folder,cur_exp)
  try: 
    os.mkdir(exp_folder)
    print("made")
  except:
    print("didnt work or dir exists")
  with open(os.path.join(exp_folder, "test_results.csv"), 'w') as file:
      writer = csv.writer(file, delimiter=' ')
      writer.writerow(["embedding", "test_loss", "test_acc"])

#   embed_options = [("glove.twitter.27B.200d", 200), ("glove.6B.50d", 50),("charngram.100d", 100)]
  for option in embed_options:
    # load fastext simple embedding with 300d
    print("doing ", option)
    embedding = option[0]
    dim = option[1]
    batch_size = 32
    num_hidden_nodes = 32
    num_layers = 2
    bidirection = True
    dropout = 0.2
    model_id = 0
    optimizer_id = 0
    criterion_id = 0
    n_epochs = 15
    learning_rate = .1

    set_up_exp(exp_folder,embedding,dim,batch_size,num_hidden_nodes,num_layers,bidirection,dropout,model_id,optimizer_id,criterion_id,n_epochs, learning_rate)

def set_up_exp(exp_folder,embedding,dim,batch_size,num_hidden_nodes,num_layers,bidirection,dropout,model_id,optimizer_id,criterion_id, N_EPOCHS, learning_rate, momentum=.99,weight_decay=1e-5):
    text_field = Field(
    sequential=True,
    tokenize='basic_english', 
    lower=True,
    include_lengths=True,
    batch_first=True,
        )

    label_field = LabelField(dtype = torch.float,batch_first=True)

    fields = [(None, None),('text',text_field),('label', label_field)]

    train, valid, test = TabularDataset.splits(path="", train='data_vectors/train-text-set-clean.tsv', validation='data_vectors/val-text-set-clean.tsv', test='data_vectors/test-text-set-clean.tsv', 
    format='TSV', fields=fields, skip_header=True)
    text_field.build_vocab(
            train, 
            vectors=embedding
        )
    label_field.build_vocab(train)

    #No. of unique tokens in text
    print("Size of TEXT vocabulary:",len(text_field.vocab))

    #No. of unique tokens in label
    print("Size of LABEL vocabulary:",len(label_field.vocab))

    #Commonly used words
    print(text_field.vocab.freqs.most_common(15))  

    #Word dictionary
    # print(text_field.vocab.stoi)  

    # get the vocab instance
    vocab = text_field.vocab

    train_iter, valid_iter, test_iter = BucketIterator.splits(
        (train, valid, test), 
        batch_size = batch_size,
        sort_key = lambda x: len(x.text),
        sort=True,
        sort_within_batch=True,
        device = device)

    size_of_vocab = len(text_field.vocab)

    num_output_nodes = 1

    if model_id == 0:
      model = basic_lstm(size_of_vocab, dim, num_hidden_nodes,num_output_nodes, num_layers, 
                      bidirectional = bidirection, dropout = dropout)
    else:
      pass
    model = model.to(device)

    pretrained_embeddings = text_field.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    if optimizer_id == 0:
      optimizer = optim.SGD(model.parameters(), lr=learning_rate) #.1
    elif optimizer_id == 1:
      optimizer = optim.Adam(model.parameters(),lr=learning_rate) # .0006
  
    if criterion_id == 0:
      criterion = nn.BCELoss()
    else:
      pass
    criterion = criterion.to(device)

    ### following code loads model and gets result for test set
    # model = basic_lstm(size_of_vocab, dim, num_hidden_nodes,num_output_nodes, num_layers, 
    #                   bidirectional = bidirection, dropout = dropout)
    # param = torch.load("word_emb_experiments/full_experiment/"+embedding+"-model")
    # model.load_state_dict(param)
    # model.eval()
    # test_loss, test_acc = evaluate(model, test_iter, criterion)
    
    # with open(os.path.join(exp_folder, "test_results.csv"), 'a') as file:
    #   writer = csv.writer(file, delimiter=' ')
    #   writer.writerow([embedding, test_loss, test_acc])
    # print("done")

    out_file = open(os.path.join(exp_folder,embedding+"-results.csv"),"wt")
    tsv_writer = csv.writer(out_file, delimiter=' ')
    tsv_writer.writerow(['train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        
        #train the model
        train_loss, train_acc = train_network(model, train_iter, optimizer, criterion)
        
        #evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iter, criterion)
        
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        tsv_writer.writerow([train_loss, train_acc, valid_loss, valid_acc])
    test_loss, test_acc = evaluate(model, test_iter, criterion)
    
    with open(os.path.join(exp_folder, "test_results.csv"), 'a') as file:
      writer = csv.writer(file, delimiter=' ')
      writer.writerow([embedding, test_loss, test_acc])
    
    torch.save(model.state_dict(), os.path.join(exp_folder, embedding+"-model.pt"))

experiment_word_emb()