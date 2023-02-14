import os
import sys
sys.path.append(os.getcwd())
import assemble as assemble

pnet_postive_file = '/disk3/mtcnn/data_set/train/12/pos_12.txt'
pnet_part_file = '/disk3/mtcnn/data_set/train/12/part_12.txt'
pnet_neg_file = '/disk3/mtcnn/data_set/train/12/neg_12.txt'
imglist_filename = '/disk3/mtcnn/anno_store/imglist_anno_12.txt'

if __name__ == '__main__':

    anno_list = []

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
