import string
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np

def compute_IOU(rec1, rec2):
    #rec: rectangle, [xmin ymin xmax ymax]
    #return IoU of rec1 and rec2

    width=max(0,min(rec1[2],rec2[2])-max(rec1[0],rec2[0]))
    hight=max(0,min(rec1[3],rec2[3])-max(rec1[1],rec2[1]))
    inter=width*hight
    union=(rec1[3]-rec1[1])*(rec1[2]-rec1[0])+(rec2[3]-rec2[1])*(rec2[2]-rec2[0])-inter
    return inter/(union+1e-8)

def get_df_iou(x):
    return compute_IOU(x[['x1_dt', 'y1_dt', 'x2_dt', 'y2_dt']].tolist(), 
                       x[['x1_gt', 'y1_gt', 'x2_gt', 'y2_gt']].tolist())

def calc_mAP(df_dt, df_gt, conf=0):
#     import pdb;pdb.set_trace()
    if len(df_dt)==0 or len(df_gt)==0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    df = pd.merge(df_dt, df_gt, on='image_id')
    if len(df)==0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    correct = 0
    for _, df_cur in df.groupby('image_id'):
        df_cur['iou'] = df_cur.apply(lambda x: get_df_iou(x), axis=1)
        df_cur = df_cur[df_cur['iou'] > 0.5]
        df_eq = df_cur[df_cur['label_gt']==df_cur['label_dt']]
        if len(df_eq)==0:
            continue
            
        # 搞两个hash对应表
        gt_box_list = np.unique(df_eq['bbox_idx_gt'].tolist()).tolist()
        dt_box_list = np.unique(df_eq['bbox_idx_dt'].tolist()).tolist()

        x, y = len(gt_box_list), len(dt_box_list)
        maxx = max(x, y) + 100 # 设定一个无法达到的数等于这个数则表示无法匹配
        matrix = np.ones((x, y)) * maxx

        for idx_box in gt_box_list:
            # 取当前行所有对应的列数据，并标注为可达
            df_eq_cur = df_eq[df_eq['bbox_idx_gt'] == idx_box]
            link_path = df_eq_cur['bbox_idx_dt'].apply(lambda x: dt_box_list.index(x)).tolist()
            matrix[gt_box_list.index(idx_box), link_path] = 1

        row_ind, col_ind = linear_sum_assignment(matrix)
        for row, col in zip(row_ind, col_ind):
            if matrix[row][col]!=maxx:
                correct+=1

    error = len(df_dt) - correct
    miss = len(df_gt) - correct
    mAP = correct/(correct + error + 1e-8)
    mAR = correct/(correct + miss + 1e-8)
    beta = 1
    mFS = (1 + beta**2)*(mAP*mAR)/((beta**2)*mAP + mAR + 1e-8)

    return correct, error, miss, mAP, mAR, mFS