import os
import pdb
import pprint
import itertools
import random
random.seed(5)


test_dir = "/home/ahmadob/dataset/facerecognition_dataset/test_set"

target_info = {}
for i in os.listdir(test_dir):
    target_info[i]={
        'path':os.path.join(test_dir, i),
        'total_image':len(os.listdir(os.path.join(test_dir, i))),
        'images_name': [os.path.join(test_dir, i, k) for k in os.listdir(os.path.join(test_dir, i))]
    }

# We expect that the images that are consequent will have high similarity so while creating pair of match we will
# select the one that are far apart
for key in target_info.keys():
    match_pair = []
    removed_images = []
    range_limit = target_info[key]['total_image']//4
    for i in range(range_limit):
        pair = [target_info[key]['images_name'][i], target_info[key]['images_name'][-range_limit+i]]
        match_pair.append(pair)
        removed_images.extend(pair)

    target_info[key]["match_pair"] = match_pair
    target_info[key]["remaining_images"] = list(set(target_info[key]['images_name']) - set(removed_images))

    # print(target_info[key]['total_image'], len(target_info[key]['remaining_images']))



### We need to process in the order of minimum technique so that there is no conflict while creating pair

final_matching_pair = []
for k in target_info.keys():
    final_matching_pair.extend(target_info[k]['match_pair'])


total_matching_images = len(final_matching_pair)
print("Total matching images = {}".format(total_matching_images))

def list_combination(list1, list2):
    result = [list(zip(x, list2)) for x in itertools.permutations(list1, len(list2))]
    return result


keyss = [k for k in target_info.keys()]

keys_combination = list(itertools.combinations(keyss, 2))
# print(keys_combination)

# from keys combination we can find out all the possible combination
# so we can divide the total number of matching pairs we have into equal number of unmatching pair
# example : we have total of 41 matching pair in 3 subjects, so unmatching pair can be 14 from each combination giving total of 42 unmatching pairs

unmatching_pair = []
n = int(round(total_matching_images / len(keys_combination)))

for k1, k2 in keys_combination:
    unmatching_pair.append(list_combination(random.sample(target_info[k1]['remaining_images'], 7), random.sample(target_info[k2]['remaining_images'], 7)))
# unmatching_pair contains total combination of 7 factorial

unmatching_pairs = []
for i in unmatching_pair:
    unmatching_pairs.extend(random.sample(i, n)) # selecting some numbers only from the 7 factorials depending on n


final_unmatching_pair = []
for i in unmatching_pairs:
    for j in i:
        final_unmatching_pair.append(j)

final_unmatching_pair = random.sample(list(set(final_unmatching_pair)), total_matching_images)

# print(final_unmatching_pair[0])
print("Total unmatching images = {}".format(len(final_unmatching_pair)))


# Saving the meta data information like pair and the corresponding 1 and 0 for matching and not matching

destination_file = 'data/custom_test_pair.txt'

if os.path.exists(destination_file):
    print("File exist, removing it for now.")
    os.remove(destination_file)
    print("File removed!")

print("Saving the file {}".format(destination_file))
for match, unmatch in zip(final_matching_pair, final_unmatching_pair):

    # unmatch_line = unmatch[0] + " " + unmatch[1] + " " + str(0)
    # print(unmatch[0])
    
    output_file = open(destination_file, 'a+')
    match_line = match[0] + " " + match[1] + " " + str(1)
    unmatch_line = unmatch[0] + " " + unmatch[1] + " " + str(0)
    output_file.write(match_line + "\n")
    output_file.write(unmatch_line + "\n")
    output_file.close()

print("File {} saved successfully".format(destination_file))




