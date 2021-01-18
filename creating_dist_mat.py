import numpy as np
import tensorflow as tf

# def create_dist_matrix_for_encdoded(encoded):
#     dist_mat = tf.ones((len(encoded),len(encoded)))
#     for base_i in range(len(encoded)):
#         base_encoded = encoded[base_i]
#         for rest_i in range(len(encoded)):
#             if rest_i > base_i:
#                 cur_encoded = encoded[rest_i]
#                 dist = tf.math.sqrt(
#                     tf.math.reduce_sum(
#                         tf.math.square(base_encoded-cur_encoded)
#                     )
#                 )
#                 dist_mat[base_i,rest_i]=dist*dist_mat[base_i,rest_i]
#
#     return dist_mat/tf.reduce_max(dist_mat)

def create_dist_matrix_for_encdoded(encoded):
    dist_mat = np.ones((len(encoded),len(encoded)))
    for base_i in range(len(encoded)):
        base_encoded = encoded[base_i]
        for rest_i in range(len(encoded)):
            cur_encoded = encoded[rest_i]
            dist = np.sqrt(
                np.sum(
                    np.square(base_encoded-cur_encoded)
                )
            )
            dist_mat[base_i,rest_i]=dist*dist_mat[base_i,rest_i]

    return dist_mat/np.max(dist_mat)

def create_squeezed_sift_matrix(big_dist_matrix,inds):
    sift_dist_matrix = np.zeros((len(inds),len(inds)))
    for i in range(len(inds)):
        for j in range(len(inds)):
            if j>i:
                sift_dist_matrix[i,j] = big_dist_matrix[inds[i],inds[j]]
    return sift_dist_matrix/np.max(sift_dist_matrix)