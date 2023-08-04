### The generation of each mat file

You can use our pre-processed datasets, which makes it easier to get start.

To generate required `.mat` files, you could:
1. Download cleaned datasets in  `pan.baidu.com`:
    
    link：https://pan.baidu.com/s/1jCYEBhm-bpikAh_Bti139g 
    password：9idm

2. Move the downloaded  `all_imgs.txt`, `all_tags.txt`, and `all_labels.txt` to `./dataset/XXXDatasetName/` as follows:
    ```
    dataset
    ├── coco
    │   ├── all_imgs.txt 
    │   ├── all_tags.txt
    │   └── all_labels.txt
    ├── flickr25k
    │   ├── all_imgs.txt 
    │   ├── all_tags.txt
    │   └── all_labels.txt
    └── nuswide
        ├── all_imgs.txt 
        ├── all_tags.txt
        └── all_labels.txt
    ```
3. Modify variable `img_root_path` in scripts `make_XXXDatasetName.py` to the absolute path of the directory, which contains all source images and is available at above provided `pan.baidu.com` link.
4. Run scripts `make_XXXDatasetName.py` to generate corresponding `.mat` files. Then use these mat files to conduct experiment.



### (Optional) The meaning and format of each mat file

#### caption.mat
For each dataset, `caption.mat` is data of text modality. It is a mat file with key `caption`.
The shape of this mat is, i.e., `(20015,)` for MIRFlickr25K. 
Each element of this mat is a `string` that 
describes one image, i.e., "cigarette tattoos smoke red dress sunglasses" for `im1.jpg` in MIRFlickr25K dataset.

Note that 20,015 instances of MIRFlickr25K with 1,386 frequent textual tags and 190,421 instances of NUSWIDE with 1,000 frequent textual tags are used for experiments.

For MS COCO, we obtain 122,218 data points by removing the pairs without any label following DCHMT, and one of five sentences is randomly selected to form one image-text pair.

#### index.mat

`index.mat` is a mat file with key `index`. The shape is `(20015,)` for MIRFlickr25K. 
Each element is a `string` that indicates image path, i.e., "/path/flickr25k/im1.jpg".

#### label.mat

`label.mat` is a mat file with key `label`. The shape is `(20015, 24)` for MIRFlickr25K. 
Each element is a `numpy.ndarray`, i.e., `[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 1. 1. 0. 0. 0. 0.]`.

For all dataset, the detailed data is showed as follows:

|   Dataset    |        File name         |    Shape     |                 One element                  |
|:------------:|:------------------------:|:------------:|:--------------------------------------------:|
| MIRFlickr25K |       caption.mat        |   (20015,)   | cigarette tattoos smoke red dress sunglasses |
| MIRFlickr25K |        index.mat         |   (20015,)   |                /path/im1.jpg                 |
| MIRFlickr25K |        label.mat         | (20015, 24)  |                [0. 0. ... 0.]                |
|   MS COCO    |       caption.mat        |  (122218,)   |   A woman cutting a large white sheet cake   |
|   MS COCO    |        index.mat         |  (122218,)   |     /path/COCO_val2014_000000522418.jpg      |
|   MS COCO    |        label.mat         | (122218, 80) |                [1. 0. ... 0.]                |
|   NUSWIDE    |       caption.mat        |  (190421,)   | portrait man flash sunglasses actor december |
|   NUSWIDE    |        index.mat         |  (190421,)   |          /path/0001_2124494179.jpg           |
|   NUSWIDE    |        label.mat         | (190421, 21) |                [0. 0. ... 0.]                |

You should generate these mat files in above format for experiments.
