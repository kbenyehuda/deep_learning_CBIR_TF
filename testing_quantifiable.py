from testing_main import *
import pickle
from glob import glob
from CBIR import return_best_idxs,show_results_func


def prec(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return len(lst3)/len(lst1)

def get_img_name_from_img_path(img_path):
    idx = img_path.index('Image')
    img_name = img_path[idx + 6:-4]
    return img_name

def get_idx_of_sift_from_img_name(image_name,all_pics):
    idx = [i for i in range(len(all_pics)) if "Image\\" + image_name + ".png" in all_pics[i]][0]
    if idx == 707:
        raise ValueError('image number 707 from glob does not exist')
    elif idx > 707:
        idx -=1
    return idx

def return_CBIR_names_from_SIFT(SIFT_dist_matrix,querry_image_name,test_filepaths,number_to_return=10,show_results=False,querry_image_test_idx=None):
    inp_path_dir = 'D:/Users/Keren/Documents/university/Year 3/deep learning/my_project/DataBase/ProjectDB/DataBaseImage/'
    all_pics = glob(inp_path_dir + '*.png')
    querry_image_sift_idx = get_idx_of_sift_from_img_name(querry_image_name,all_pics)
    best_idxs = return_best_idxs(querry_image_sift_idx,SIFT_dist_matrix,3062)

    number_of_returns = 0
    i=0
    best_imgs_indexes_in_test = []
    while number_of_returns < number_to_return:
        real_idx = best_idxs[i + 1] + (best_idxs[i]>706)*1
        next_closest_img_name = get_img_name_from_img_path(all_pics[real_idx])
        idx_in_test_filepaths = [i for i in range(len(test_filepaths)) if "Image/" + next_closest_img_name + ".png" in test_filepaths[i]]
        if len(idx_in_test_filepaths) > 0:
            number_of_returns +=1
            best_imgs_indexes_in_test.append(idx_in_test_filepaths[0])
        i+=1

    if show_results:
        show_results_func(test_filepaths,best_imgs_indexes_in_test,querry_image_test_idx)

    return best_imgs_indexes_in_test


def return_prec(querry_image_test_idx,number_to_return,show_results = False):
    # querry_image_test_idx = 56
    querry_image_name = get_img_name_from_img_path(pca_filepaths[querry_image_test_idx][50:])
    # number_to_return = 10

    best_idxs_300 = return_CBIR(querry_image_test_idx, dist_300, number_to_return + 1, pca_filepaths, return_idxs=True,
                                show_results=show_results)
    best_idxs_300_5 = return_CBIR(querry_image_test_idx, dist_300_5, number_to_return + 1, pca_filepaths,
                                  return_idxs=True, show_results=show_results)
    best_idxs_50 = return_CBIR(querry_image_test_idx, dist_50, number_to_return + 1, pca_filepaths, return_idxs=True,
                               show_results=show_results)
    best_idxs_5 = return_CBIR(querry_image_test_idx, dist_5, number_to_return + 1, pca_filepaths, return_idxs=True,
                              show_results=show_results)

    best_idxs_from_sift = return_CBIR_names_from_SIFT(SIFT_dist, querry_image_name, pca_filepaths, number_to_return,
                                                      show_results=show_results,querry_image_test_idx=querry_image_test_idx)

    prec_300 = prec(best_idxs_300, best_idxs_from_sift)
    prec_300_5 = prec(best_idxs_300_5, best_idxs_from_sift)
    prec_50 = prec(best_idxs_50, best_idxs_from_sift)
    prec_5 = prec(best_idxs_5, best_idxs_from_sift)

    return prec_300,prec_300_5,prec_50,prec_5


if __name__=='__main__':

    SIFT_dist_dir = 'D:/Users/Keren/Documents/university/Year 3/deep learning/'
    pickle_in = open(SIFT_dist_dir + 'dists_matrix.pickle','rb')
    dic = pickle.load(pickle_in)
    SIFT_dist = dic['dist_matrix']

    train_loss,val_loss,test_loss,test_output_300, test_output_300_5, test_output_50, test_output_5, pca_test, pca_filepaths = testing('20210108-180603','0083',encoder)
    dist_300 = create_dist_matrix_for_encdoded(test_output_300)
    dist_300_5 = create_dist_matrix_for_encdoded(test_output_300_5)
    dist_50 = create_dist_matrix_for_encdoded(test_output_50)
    dist_5 = create_dist_matrix_for_encdoded(test_output_5)


    prec_300 = 0
    prec_300_5 = 0
    prec_50 = 0
    prec_5 = 0
    for i in range(6,7):
        precs = return_prec(i,10,show_results=True)
        # prec_300 += precs[0]
        # prec_300_5 += precs[1]
        # prec_50 += precs[2]
        # prec_5 += precs[3]

    prec_300 /= len(pca_filepaths)
    prec_300_5 /= len(pca_filepaths)
    prec_50 /= len(pca_filepaths)
    prec_5 /= len(pca_filepaths)
