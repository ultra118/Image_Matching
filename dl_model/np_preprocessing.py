
# coding: utf-8

import numpy as np

def np_append(np_1,np_2, class_dict,np_shape=200):
    np_1 = np.zeros((1,np_shape,np_shape,3))
    np_2 = np.zeros((1,np_shape,np_shape,3))
    for k in class_dict.keys():
        for i, arr_list in enumerate(class_dict[k]):
            if i % 20 == 0:
                print('{} : {} / {} ...'.format(k, i, len(class_dict[k])))
            for i_, np_arr in enumerate(arr_list):
                img_ = np_arr.reshape((-1,np_shape,np_shape,3))
                if i_ == 0:
                    np_1 = np.append(np_1, img_, axis = 0)
                else:
                    np_2 = np.append(np_2, img_, axis = 0)
    return np_1, np_2