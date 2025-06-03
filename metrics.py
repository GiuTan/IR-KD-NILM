import numpy as np
import tensorflow as tf
from tensorflow.keras  import backend as K
# The energy error is the difference between
# the total predicted energy, and the total actual energy consumed
# by each active appliance in that sample instant


def TECA(Y_pred, Y_test, dataset, classes=0):
    classes = classes

    if dataset=='REFIT':
        kettle_val = 2066 # len 23.67 samples
        micro_val = 960.51 # len 24.88 samples
        wash_val = 300.50 # len 918.25 samples
        dish_val = 597.48 # len 970.43 samples
        toa_val = 1148 # len 19.28 samples
        washer_val = 1060.57 # len 304.5 samples (presente solo in una casa di train!)
    else: #ukdale
        kettle_val = 1968 # 21.57 samples
        micro_val = 969.07 # len 19.19 samples
        wash_val = 511.86 # 857.84 samples
        dish_val = 802.45 # 1059.74 samples
        toa_val = 1437 # len 33.87 samples
        washer_val = 504.28 # len 849.36 samples

    mean_val = [kettle_val, micro_val, wash_val, dish_val, toa_val, washer_val] #[kettle_val, micro_val, wash_val, dish_val, toa_val, washer_val]
    y_ave = Y_pred
    gt = Y_test
    y_ave_t = y_ave.transpose()
    gt_t = gt.transpose()
    tot_sum = 0.0
    for k in range(classes):
        if classes == 6:
            if (k == 2 or k == 3 or k == 5): #prendo solo wash dish e washer
                y_ave_t[k] = y_ave_t[k] * mean_val[k]
                gt_t[k] = gt_t[k] * mean_val[k]

                diff = np.sum(np.abs(gt_t[k]-y_ave_t[k]))

                tot_sum = tot_sum + diff
        else:
            y_ave_t[k] = y_ave_t[k] * mean_val[k]
            gt_t[k] = gt_t[k] * mean_val[k]

            diff = np.sum(np.abs(gt_t[k] - y_ave_t[k]))

            tot_sum = tot_sum + diff
    num = tot_sum
    den = 2 * np.sum(gt_t)

    TECA_ = num / den
    TECA_ = 1-TECA_


    return TECA_
