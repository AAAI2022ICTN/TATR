This is the code for our paper "Topic-Aware Tag Recommendation for Textual Content"  
# Requirements:  
Python 3.7.10;  
Pytorch 1.8.1;  

# Data  

## Law  
We provide the processed dataset Law.  
Processed data download: https://drive.google.com/drive/folders/10udaQnfoGOHWIFFAjcNFUTRzfNGG3yPZ?usp=sharing  
Because other processed datasets are too large, we provide the original version and processing method of the data. Processing method can be found in /processing/.  

## Twitter  
Data: https://drive.google.com/drive/folders/11OyK5X73g3hw-FLiwmp09zt7xmRtm2GH?usp=sharing  

## Physics  
Data: https://ia600107.us.archive.org/view_archive.php?archive=/27/items/stackexchange/physics.stackexchange.com.7z  
(Use Posts.xml)  

## AU
Data: https://ia800107.us.archive.org/view_archive.php?archive=/27/items/stackexchange/askubuntu.com.7z  
(Use Posts.xml)  

# Reproducibility:  

Take dataset Law as an example.  
## configuration file  
Please confirm the corresponding configuration file:  
```bash
myConfig.py
```

## Step 1: NTM  
We firstly only train NTM to get the pre-trained NTM by:  
```bash
python main.py  -load_path law/ -only_train_ntm -topic_num 300  
```
/law/ is the DATA PATH, 300 is the topic number.  
Pre-trained NTM PATH:
model/AU.topic_num300.ntm_warm_up_100.20210630-112822/e100.val_loss=414.776.sparsity=0.700.ntm_model

## Step 2: 
To train Feature Extracor and NTM jointly, and report evaluation by:
```bash
python main.py -load_path law/ -use_topic_represent -load_pretrain_ntm -joint_train -topic_num 300  -attn_mode MTA2 -check_pt_ntm_model_path model/AU.topic_num300.ntm_warm_up_100.20210630-112822/e100.val_loss=414.776.sparsity=0.700.ntm_model
```
MTA2 is the Topic-Aware Attention Mechanism we reported in our paper.  
