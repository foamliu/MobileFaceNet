import os
import pdb


def read_file(file_path):
    print("Reading file : {}".format(file_path))
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [ele.strip() for ele in lines] # remove \n from each string and blank lines
    lines = [ele for ele in lines if ele.strip()] # removing empty string
    return lines

def checker(lines):
    # reads the line of files, checks their first name and then puts 1 or 0 based on their match
    new_list = []
    for i in range(len(lines)-1):
        if i % 2 == 0:
            match = int(lines[i].split('/')[0] == lines[i + 1].split('/')[0])
            new_list.append(lines[i] + " "+ lines[i+1] +" " + str(match))
    return new_list


def save_file(lines):
    print("Saving the file")
    output_file = open(destination_file, 'a+')
    for line in lines:
        output_file.write(line + "\n")
    output_file.close()

    print("File saved successfully")


def test_read_saved_file(file_path):
    print("Reading saved file")
    with open(file_path, 'r') as file:
        lines = file.readlines()

    print("Total items in file are = {}".format(len(lines))) # total of 6000 is printed after successfully running the code

if __name__ == "__main__":
    source_folder = 'data/lfw_test_pair' # contains the files pairs_01.txt, pairs_02.txt, .... pairs_10.txt
    destination_file = 'data/lfw_test_pair.txt'

    if os.path.exists(destination_file):
        print("File exist, removing it for now.")
        os.remove(destination_file)
        print("File removed!")

    list_of_files = os.listdir(source_folder)

    full_lines = [] # for holding all the files list
    for file in list_of_files:
        file_path = os.path.join(source_folder, file)
        lines = read_file(file_path)
        new_lines = checker(lines)
        full_lines.extend(new_lines)

    save_file(full_lines)
    test_read_saved_file(destination_file)


