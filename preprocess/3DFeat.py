import numpy as np
import os
import glob
from tqdm import tqdm
import datetime
import time

def Caltime(date1,date2):

    date1=time.strptime(date1,"%Y%m%d")
    date2=time.strptime(date2,"%Y%m%d")

    date1=datetime.datetime(date1[0],date1[1],date1[2])
    date2=datetime.datetime(date2[0],date2[1],date2[2])

    return date2-date1

# d=Caltime('20180901','20181001')
def get_txt_feat(txt_path,feat_save_dir):
    with open(txt_path,'r') as f:
        lines_list=[l.strip().split('\t')[1].split(',') for l in f.readlines()]
    people_num=len(lines_list)

    item_list=[]
    for l in lines_list:
        item_list+=l

    feat_cal=np.zeros((26,7,24))
    for item in item_list:
        ymd_t=item.split('&')
        ymd=ymd_t[0]
        
        ymd_st='20181001'
        delta_day=Caltime(ymd_st,ymd).days

        week_num=int(delta_day/7)
        weekday=int(datetime.datetime.strptime(ymd,"%Y%m%d").weekday())

        h_list=ymd_t[1].split('|')
        h_int=[int(h) for h in h_list]
        for h in h_int:
            feat_cal[week_num,weekday,h]+=1
        
    feat_cal=feat_cal*1.0/people_num
    save_path=feat_save_dir+os.path.basename(txt_path).split('.')[0]+'.npy'

    np.save(save_path,feat_cal)

def batch_get_feat(in_dir,feat_save_dir):
    txt_path_list=glob.glob(in_dir+'*.txt')

    for txt_path in tqdm(txt_path_list,ascii=True):
        get_txt_feat(txt_path,feat_save_dir)

if __name__=='__main__':
    in_dir='data/train/'
    out_dir='data/train_txt3dfeat/'
    batch_get_feat(in_dir,out_dir)
    print('done')