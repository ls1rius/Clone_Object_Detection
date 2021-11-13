import string
import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np

def compute_IOU(rec1,rec2):
    #rec: rectangle, [xmin ymin xmax ymax]
    #return IoU of rec1 and rec2
    if type(rec1) is str:
        rec1 = [float(item) for item in rec1[1:-1].split(',')]
    else:
        rec1 = [float(item) for item in rec1]

    if type(rec2) is str:
        rec2 = [float(item) for item in rec2[1:-1].split(',')]
    else:
        rec2 = [float(item) for item in rec2]

    width=max(0,min(rec1[2],rec2[2])-max(rec1[0],rec2[0]))
    hight=max(0,min(rec1[3],rec2[3])-max(rec1[1],rec2[1]))
    inter=width*hight
    union=(rec1[3]-rec1[1])*(rec1[2]-rec1[0])+(rec2[3]-rec2[1])*(rec2[2]-rec2[0])-inter
    return inter/(union+1e-8)

def get_df_iou(x):
    return compute_IOU(x['bbox'], x['gt_bbox'])

def calc_mAP(df_dt, df_gt):
    if len(df_dt)==0 or len(df_gt)==0:
        return 0, 0, 0, 0, 0, 0
    
    df = pd.merge(df_dt, df_gt, on='image_id')
    if len(df)==0:
        return 0, 0, 0, 0, 0, 0
    
    correct = 0
    for _, df_cur in df.groupby('image_id'):
        df_cur['iou'] = df_cur.apply(lambda x: get_df_iou(x), axis=1)
        df_cur = df_cur[df_cur['iou'] > 0.5]
        df_eq = df_cur[df_cur['category_id']==df_cur['label']]
        if len(df_eq)==0:
            continue
            
        gt_box_list = np.unique(df_eq['bbox_idx'].tolist()).tolist()
        dt_box_list = np.unique(df_eq['dt_bbox_idx'].tolist()).tolist()

        x, y = len(gt_box_list), len(dt_box_list)
        maxx = max(x, y) + 100
        matrix = np.ones((x, y)) * maxx

        for idx_box in gt_box_list:
            df_eq_cur = df_eq[df_eq['bbox_idx'] == idx_box]
            link_path = df_eq_cur['dt_bbox_idx'].apply(lambda x: dt_box_list.index(x)).tolist()
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