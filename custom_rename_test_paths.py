import os


source_filename = 'data/custom_test_pair.txt'
destination_filename = 'data/custom_test_face_pair.txt'
with open(source_filename, 'r') as file:
    pair_lines = file.readlines()

output_file = open(destination_filename, 'a+')

for i in pair_lines:
    pths = i.split(' ')
    line = pths[0].replace('test_set', "test_face_set") + " " + pths[1].replace('test_set', "test_face_set") + " "+pths[2]
    output_file.write(line)
