# generate mat data for MSCOCO
import os
import scipy.io as scio
import numpy as np
import random
import torch
from sklearn.feature_extraction.text import CountVectorizer

seed = 1
torch.random.manual_seed(seed=seed)
np.random.seed(seed=seed)
random.seed(seed)


img_path = 'all_imgs.txt'
txt_path = 'all_txts.txt'
lab_path = 'all_labels.txt'


# TODO 修改为包含所有图片(约120K张,包括训练集和验证集)的文件夹的绝对路径
img_root_path = "/TO/you/path/of/coco_all_images"

'''
####################### sentences #######################
'''


def sentence2str(f_name):
    # return python dict: {key=i, value=""}
    with open(f_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    res = {}
    for i, line in enumerate(lines):
        line = line.strip()
        sentences_5 = line.split(';')
        use_cap = sentences_5[random.randint(0, len(sentences_5) - 1)]  # use random one
        use_cap = use_cap.split(".")[0]  # remove "."
        res[i] = use_cap
    return res


sentences_strs_dict = sentence2str(txt_path)  # one random sentence


'''
####################### labels #######################
'''
labels = np.loadtxt(lab_path, dtype=np.float32)  # ndarray: (N, 80)
labels_dict = {}
for i in range(labels.shape[0]):
    labels_dict[i] = labels[i]


'''
####################### image paths #######################
'''
img_abs_paths_dict = {}  # python dict {key=id, value='**/COCO_val2014_000000522418.jpg}'
with open(img_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        img_abs_paths_dict[i] = os.path.join(img_root_path, line.strip().split("/")[-1])


'''
####################### remove empty label (all 0) #######################
all_labels.txt contains 1069 empty labels (all 0).
Following DCHMT (Differentiable Cross Modal Hashing via Multimodal Transformers), we remove them for experiment.
'''

used_keys = []  # list
for key in labels_dict:
    if sum(labels_dict[key]) != 0:
        used_keys.append(key)


sentences_strs_list = []
img_abs_paths_list = []
labels_list = []
for key in used_keys:  # 122218 instances remained
    sentences_strs_list.append(sentences_strs_dict[key])
    img_abs_paths_list.append(img_abs_paths_dict[key])
    labels_list.append(labels_dict[key])


txt = np.asarray(sentences_strs_list)
img = np.asarray(img_abs_paths_list)
lab = np.asarray(labels_list)


'''
####################### sentences to one hot(BoW) #######################
'''
# https://blog.csdn.net/Robin_Pi/article/details/103599437


def sentences2bow(text):
    print("---------------------------------Generate BoW---------------------------------")
    vectorizer_1 = CountVectorizer()
    vectorizer_1.fit(text)
    word2id = vectorizer_1.vocabulary_  # K 是单词, V 是单词id，不是出现次数
    id2word = {}
    for k in word2id:
        id = word2id[k]
        id2word[id] = k

    vector = vectorizer_1.transform(text)  # <class 'scipy.sparse.csr.csr_matrix'>
    bows_all_words = vector.toarray()  # ndarray
    print(bows_all_words.shape)

    # 根据不同的sklearn版本选择二者其一:
    # feature_names = vectorizer_1.get_feature_names_out()
    feature_names = vectorizer_1.get_feature_names()

    print('original num_of_words: {}'.format(len(feature_names)))  # 14464

    # using torch to process 2000dim BoW.
    bows_all_words = torch.from_numpy(bows_all_words)  # ND
    words_cnt = bows_all_words.sum(dim=0)  # 出现次数,D
    val, idx = torch.topk(words_cnt, k=2000)  # 2000
    
    top2k_vocabulary = []
    for idxi in idx.numpy():
        top2k_vocabulary.append(id2word[idxi])

    # select top 2K words
    vectorizer_2 = CountVectorizer(vocabulary=top2k_vocabulary)
    vectorizer_2.fit(text)
    vector = vectorizer_2.transform(text)
    bow = vector.toarray()

    # feature_names = vectorizer_2.get_feature_names_out()
    feature_names = vectorizer_2.get_feature_names()
    
    print('After process: num_of_features:{}'.format(len(feature_names)))  # 14464

    return bow


txt_one_hot = sentences2bow(txt)


'''
####################### output #######################
'''
print("--------------------------------------------------------------------")
print("txts: ", txt.shape)
print("0th txt: ", txt[0])
print("--------------------------------------------------------------------")
print("imgs: ", img.shape)
print("0th img: ", img[0])
print("--------------------------------------------------------------------")
print("labels: ", lab.shape)
print("0th label: ", lab[0])
print("--------------------------------------------------------------------")
print("txts one hot: ", txt_one_hot.shape)
print("0th txt: ", txt_one_hot[0])


# save path
root_dir = "."


# 处理成ndarray的字典，保存
scio.savemat(os.path.join(root_dir, "index.mat"), {'index': img})  # ndarray: (20015,)
scio.savemat(os.path.join(root_dir, "caption.mat"), {'caption': txt})  # ndarray: (20015,)
scio.savemat(os.path.join(root_dir, "caption_one_hot.mat"), {'caption_one_hot': txt_one_hot})  #   # ndarray: (20015, 1386)
scio.savemat(os.path.join(root_dir, "label.mat"), {'label': lab})  # ndarray: (20015, 24)
print("Save all *.mat")

