"""
This code goes through the overall custom dataset directory.
Select 90% of images for each subject for training
Select remaining 10% of images for testing

Then create matching and non-matching sample pair from test dataset
"""
import os
import pdb
import pprint
import itertools
from itertools import combinations, product
import random
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
random.seed(5)


def data_splitter(overall_images_list):
    train_list, test_list = train_test_split(overall_images_list, test_size=0.10, random_state=42, shuffle=True)
    return train_list, test_list

def save_data(target_info, kind='train_list'):
    if kind == "train_list":
        root_train_dir = "/home/ahmadob/dataset/facerecognition_dataset/train_set"
    if kind == "test_list":
        root_train_dir = "/home/ahmadob/dataset/facerecognition_dataset/test_set"

    for name in tqdm(target_info.keys()):
        train_dir = os.path.join(root_train_dir, name)
        os.makedirs(train_dir, exist_ok=True)
        for src_image_path in target_info[name][kind]:
            shutil.copy(src_image_path, train_dir)


def generate_matching_pairs(target_info):
    matching_pairs = []
    for target in target_info.keys():
        test_list = target_info[target]['test_list']
        comb = combinations(test_list, 2)
        for i in list(comb):
            matching_pairs.append(i)

    return matching_pairs

def key_combinations(target_info):
    keys_comb = combinations(target_info.keys(), 2)
    key_combinations = []
    for i in list(keys_comb):
        key_combinations.append(i)
    return key_combinations

def comb_two_list(list_1, list_2):

    total_return_pair = 2
    returning_pair = []

    for i in range(total_return_pair):
        random_index_1 = random.randint(0, len(list_1) - 1)
        random_index_2 = random.randint(0, len(list_2) - 1)
        t1, t2 = list_1[random_index_1], list_2[random_index_2]

        returning_pair.append((t1, t2))
    return returning_pair
def generate_unmatching_pairs(target_info):
    key_comb = key_combinations(target_info)
    unmatching_pairs = []
    for (key1, key2) in key_comb:
        list_1, list_2 = target_info[key1]['test_list'], target_info[key2]['test_list']
        unmatching_pairs.extend(comb_two_list(list_1, list_2))

    return unmatching_pairs


def change_path_for_test_set(target_info):
    for key in target_info.keys():
        for pth_index in range(len(target_info[key]['test_list'])):
            target_info[key]['test_list'][pth_index] = target_info[key]['test_list'][pth_index].replace("overall_jpeg_data", "test_set")
    return target_info


def make_string(pairs, kind='match'):
    return_list = []
    if kind == 'match':
        k = 1
    if kind == 'unmatch':
        k = 0
    for i in pairs:
        pair = i[0] + " " + i[1] + " "+ str(k)
        return_list.append(pair)
    return return_list


def save_text_file(matching_pairs, unmatching_pairs):
    destination_file = 'data/custom_test_pair.txt'

    if os.path.exists(destination_file):
        print("File exist, removing it for now.")
        os.remove(destination_file)
        print("File removed!")

    match_list_string = make_string(matching_pairs, kind='match')
    unmatch_list_string = make_string(unmatching_pairs, kind='unmatch')

    final_list = match_list_string + unmatch_list_string
    print("Total length of test pairs = {}".format(len(final_list)))
    random.shuffle(final_list)
    for i in final_list:
        output_file = open(destination_file, 'a+')
        output_file.write(i + "\n")



if __name__ =="__main__":

    overall_dir = "/home/ahmadob/dataset/facerecognition_dataset/overall_jpeg_data"

    target_info = {}

    for i in os.listdir(overall_dir):
        target_info[i] = {
            'path': os.path.join(overall_dir, i),
            'total_image': len(os.listdir(os.path.join(overall_dir, i))),
            'images_name': [os.path.join(overall_dir, i, k) for k in os.listdir(os.path.join(overall_dir, i))],
        }

    for individual in target_info.keys():
        target_info[individual]["train_list"], target_info[individual]['test_list'] = data_splitter(target_info[individual]['images_name'])


    #### Saving the images
    save_data(target_info, kind='train_list') # saving the training images in train_set folder
    save_data(target_info, kind='test_list')  # saving the test images in test_set folder

    target_info = change_path_for_test_set(target_info) # this is required because we need to create the test_path
    matching_pairs = list(set(generate_matching_pairs(target_info)))
    print("Total matching pairs = {}".format(len(matching_pairs)))
    unmatching_pairs = list(set(generate_unmatching_pairs(target_info)))
    print("Total unmatching pairs = {}".format(len(unmatching_pairs)))

    save_text_file(matching_pairs, unmatching_pairs)






