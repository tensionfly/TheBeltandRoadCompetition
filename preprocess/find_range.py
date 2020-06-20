import numpy as np
import os
import glob
from tqdm import tqdm

def get_txt_feat(txt_path,range_np):
    with open(txt_path,'r') as f:
        lines_list=[l.strip().split('\t')[1].split(',') for l in f.readlines()]

    item_list=[]
    for l in lines_list:
        item_list+=l

    for item in item_list:
        ymd_t=item.split('&')

        if ymd_t[0][:4] not in ['2018','2019']:
            print('erro!')

        y=int(ymd_t[0][:4]=='2019')
        md=ymd_t[0][-4:]
        m_int=int(md[:2])-1

        if m_int<range_np[y,0]:
            range_np[y,0]=m_int
        if m_int>range_np[y,1]:
            range_np[y,1]=m_int

def batch_get_feat(in_dir):
    txt_path_list=glob.glob(in_dir+'*.txt')
    range_np=np.array([[11,0],[11,0]])

    for txt_path in tqdm(txt_path_list,ascii=True):
        get_txt_feat(txt_path,range_np)

    print(range_np)

if __name__=='__main__':
    batch_get_feat('data/test/')