import numpy as np
def create_array_of_sift_feats(all_sift_feats_dic):
    all_pic_names = list(all_sift_feats_dic.keys())
    array_of_sift_feats = []
    for i in range(len(all_pic_names)):
        cur_pic_dict = all_sift_feats_dic[all_pic_names[i]]
        all_sift_vals = list(cur_pic_dict.values())
        one_vec = []
        for j in range(len(all_sift_vals)):
            one_vec = one_vec + all_sift_vals[j]
        array_of_sift_feats.append(one_vec)
    return all_pic_names,np.array(array_of_sift_feats)


def create_array_of_sift_feats_per_pic(cur_pic_dict):
    # cur_pic_dict = all_sift_feats_dic[all_pic_names[i]]
    all_sift_vals = list(cur_pic_dict.values())
    one_vec = []
    for j in range(len(all_sift_vals)):
        one_vec = one_vec + all_sift_vals[j]

    return one_vec
