# generate mat data for nuswide
import os
import scipy.io as scio
import numpy as np

img_path = 'all_imgs.txt'
txt_path = 'all_tags.txt'
lab_path = 'all_labels.txt'

tag_list_filename = 'Concepts/TagList1k.txt'


# TODO 修改为包含所有图片(约260K)的文件夹的绝对路径
img_root_path = "/TO/you/path/of/nuswide_all_images"

'''
####################### tags 1000 #######################
'''
with open(tag_list_filename, 'r', encoding='utf-8') as f:
    tags_list = f.readlines()
    tags_list = list([one.strip() for one in tags_list])


def tag2onehot(tag_filename):
    """
    # return ndarray (N, 1000) one hot, np.float32
    """
    with open(tag_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    onehot = np.zeros((len(lines), len(tags_list)), dtype=np.float32)
    for i, line in enumerate(lines):
        line = line.strip()
        tags = line.split(',')
        tags_idx = [tags_list.index(t) for t in tags]

        onehot[i, np.array(tags_idx, dtype=np.int32)] = 1
    return onehot


def tag2str(tag_filename):
    """
    # return ndarray : (N, ), each one is a str of tags, i.e., 'cigarette tattoos smoke red dress sunglasses'
    """
    with open(tag_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    res = []
    for i, line in enumerate(lines):
        line = line.strip()

        if line == "":
            print("Error: Empty tags...")
            exit()

        tags = line.split(',')
        sss = " ".join(tags)
        res.append(sss)

    return np.array(res)


tags_one_hot = tag2onehot(txt_path) # ndarray: (N, 1000)
print("--------------------------------------------------------------------")
print("tags one hot: ", tags_one_hot.shape)
print("0th tags: ", tags_one_hot[0])

tags_strs = tag2str(txt_path)  # ndarray: (N,)
print("--------------------------------------------------------------------")
print("tags str: ", tags_strs.shape)
print("0th tags: ", tags_strs[0])


'''
####################### labels 21 #######################
'''
labels = np.loadtxt(lab_path, dtype=np.float32)  # ndarray: (N, 21)
print("--------------------------------------------------------------------")
print("labels: ", labels.shape)
print("0th label: ", labels[0])


'''
####################### image paths #######################
'''
img_abs_paths = []  # python list (N), each one is str 'mirflickr/im1.jpg'
with open(img_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

    for line in lines:
        ttt = line.strip()
        img_abs_paths.append(os.path.join(img_root_path, ttt.split("/")[1]))  # abs path.

img_abs_paths = np.array(img_abs_paths)  # ndarray: (N,)
print("--------------------------------------------------------------------")
print("image paths: ", img_abs_paths.shape)
print("0th image path: ", img_abs_paths[0])


'''
####################### Save mat #######################
'''
# save path
root_dir = "."
print("--------------------------------------------------------------------")
scio.savemat(os.path.join(root_dir, "index.mat"), {'index': img_abs_paths}) 
scio.savemat(os.path.join(root_dir, "caption.mat"), {'caption': tags_strs}) 
scio.savemat(os.path.join(root_dir, "caption_one_hot.mat"), {'caption_one_hot': tags_one_hot})  
scio.savemat(os.path.join(root_dir, "label.mat"), {'label': labels}) 
print("Save all *.mat")

