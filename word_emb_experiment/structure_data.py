import json 
import pathlib
import os
import cv2
import numpy as np
from tempfile import TemporaryFile
import csv

data_directory = open('data_dir.txt', 'r').readline().strip()
# print(data_directory, os.path.isdir(data_directory))

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def find_rec(all_recs,rec_id):
    for rec in all_recs:
        if rec['id'] == rec_id:
            return rec

def format_img(path,height,width):
    img = cv2.imread(path)
    img = cv2.resize(img, (height, width))
    img = img.astype(np.float32)/255
    return img

def return_img_array(records):
    imgs = []
    for rec in records:
        img = format_img(os.path.join(data_directory, rec['img']),128,128)
        imgs.append(img)
        print(rec['id'],len(imgs))
    return imgs

def save_np_imgs(dataset, name):
    imgs = np.array(return_img_array(dataset))
    print(name, " length", imgs.shape)
    np.save(name + "-images", imgs)
    del imgs

def save_text(dataset, name):
    print(name)
    out_file = open(name+"-text.tsv","wt")
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['text', 'category'])
    for record in dataset:
        try:
            tsv_writer.writerow([record['text'],record['label']])
        except:
            tsv_writer.writerow([record['text']])

def save_data():

    train_data = load_jsonl(os.path.join(data_directory, "train.jsonl"))

    dev_seen_data = load_jsonl(os.path.join(data_directory, "dev_seen.jsonl"))
    dev_unseen_data = load_jsonl(os.path.join(data_directory, "dev_unseen.jsonl"))
    dev_data = dev_seen_data
    for rec in dev_unseen_data:
        if rec not in dev_data:
            dev_data.append(rec)
    print(len(dev_data))

    test_seen_data = load_jsonl(os.path.join(data_directory, "test_seen.jsonl"))
    test_unseen_data = load_jsonl(os.path.join(data_directory, "test_unseen.jsonl"))
    test_data = test_seen_data + test_unseen_data

    save_text(train_data, "train")
    save_text(dev_data, "dev")
    save_text(test_data, "test")

    save_np_imgs(train_data, "train")
    save_np_imgs(dev_data, "dev")
    save_np_imgs(test_data, "test")


def find_duplicates():
    print(len(train_data), len(dev_seen_data),len(dev_unseen_data),len(test_seen_data),len(test_unseen_data))
    print(len(train_data)+len(dev_seen_data)+len(dev_unseen_data)+len(test_seen_data)+len(test_unseen_data))

    all_data = [train_data,dev_seen_data,dev_unseen_data,test_seen_data,test_unseen_data]

    full_list = {}
    for i  in range(0,len(all_data)):
        for rec in all_data[i]:
            img_path = os.path.join(data_di,rec['img'])
            if not os.path.isfile(img_path):
                print(i, rec)
            if rec['id'] not in full_list:
                full_list[rec['id']] = i
            # else:
                # print(i, full_list[rec['id']])
                # print("already listed: ", i, rec, full_list[rec['id']])
                # print("first listed: ", find_rec(all_data[full_list[rec['id']]],rec['id']))
                # print(rec == find_rec(all_data[full_list[rec['id']]],rec['id']) )

    print(len(full_list))

# save_data()

# the original actually removes way too much from the text
def clean_text(text):
  text = text.translate(str.maketrans('', '', punctuation))
  text = text.lower().strip()
  text = ' '.join([i if i.isalpha() or i.isnumeric() else '' for i in text.lower().split()])
  # text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
  text = re.sub(r"\s{2,}", " ", text)
  return text

def clean_text_original(text):
  text = text.translate(str.maketrans('', '', punctuation))
  text = text.lower().strip()
  text = ' '.join([i if i not in stop and i.isalpha() else '' for i in text.lower().split()])
  text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
  text = re.sub(r"\s{2,}", " ", text)
  return text

# todo: need earlier function to comdbine test, dev datasets
def assign_data_sets():
  all_images = np.load("all-images.npy")
  print(all_images.shape)

  num_examples = all_images.shape[0]
  num_val_set = int(num_examples * .15)

  indices = np.random.permutation(num_examples)
  test_idx, val_idx, train_idx = indices[:num_val_set], indices[num_val_set:num_val_set*2], indices[num_val_set*2:]
  test_im, val_im, train_im = all_images[test_idx,:], all_images[val_idx,:], all_images[train_idx,:]

  # b text data

  text_and_label = pd.read_csv("all-text.tsv",sep='\t')
  labels = text_and_label['category'].apply(lambda cat: int(cat))
  labels = labels.to_numpy()
  test_labels, val_labels, train_labels = labels[test_idx], labels[val_idx], labels[train_idx]

  # text = text_and_label['text']
  # text_pd = text.apply(lambda text: clean_text(text))
  # text = text_pd.to_numpy()
  # test_text, val_text, train_text = text[test_idx], text[val_idx], text[train_idx]

  test, val, train = text_and_label.iloc[test_idx], text_and_label.iloc[val_idx], text_and_label.iloc[train_idx]
  test.to_csv('test-text-set.tsv',sep='\t')
  val.to_csv('val-text-set.tsv',sep='\t')
  train.to_csv('train-text-set.tsv',sep='\t')

  text_and_label['text'] = text_and_label['text'].apply(lambda text: clean_text(text))
  test, val, train = text_and_label.iloc[test_idx], text_and_label.iloc[val_idx], text_and_label.iloc[train_idx]
  test.to_csv('test-text-clean-set.tsv',sep='\t')
  val.to_csv('val-text-clean-set.tsv',sep='\t')
  train.to_csv('train-text-clean-set.tsv',sep='\t')


def clean_all_text():
  for text in ["test-text-set","val-text-set","train-text-set"]:
    text_and_label = pd.read_csv(text+".tsv",sep='\t')
    del text_and_label['Unnamed: 0']
    del text_and_label['Unnamed: 0.1']
    text_and_label['text'] = text_and_label['text'].apply(lambda text: clean_text(text))
    text_and_label.to_csv(text+"-clean.tsv",sep="\t")