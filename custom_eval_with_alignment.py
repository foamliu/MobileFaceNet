import math
import os
import pickle
import tarfile
import time

import cv2 as cv
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from facenet_utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes, ensure_folder

angles_file = 'data/custom_angles.txt'
custom_pickle = 'data/custom_test.pkl'
transformer = data_transforms['val']


def extract(filename):
    with tarfile.open(filename, 'r') as tar:
        tar.extractall('data')


def process():
    testset_dir = "/home/ahmadob/dataset/facerecognition_dataset/test_set"
    subjects = [d for d in os.listdir(testset_dir) if os.path.isdir(os.path.join(testset_dir, d))]
    assert (len(subjects) == 5), "Number of subjects is: {}!".format(len(subjects))


    print('Collecting file names...')
    file_names = []
    for i in tqdm(range(len(subjects))):
        sub = subjects[i]
        folder = os.path.join(testset_dir, sub)
        files = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]
        for file in files:
            filename = os.path.join(folder, file)
            file_names.append({'filename': filename, 'class_id': i, 'subject': sub})

    # assert (len(file_names) == 13233), "Number of files is: {}!".format(len(file_names))

    print('Aligning faces...')
    samples = []
    for item in tqdm(file_names):
        filename = item['filename']
        class_id = item['class_id']
        sub = item['subject']
        is_valid, bounding_boxes, landmarks = get_central_face_attributes(filename)

        if is_valid:
            samples.append(
                {'class_id': class_id, 'subject': sub, 'full_path': filename, 'bounding_boxes': bounding_boxes,
                 'landmarks': landmarks})

    with open(custom_pickle, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)


def get_image(samples, file):
    filtered = [sample for sample in samples if file in sample['full_path'].replace('\\', '/')]
    assert (len(filtered) == 1), 'len(filtered): {} file:{}'.format(len(filtered), file)
    sample = filtered[0]
    full_path = sample['full_path']
    landmarks = sample['landmarks']
    img = align_face(full_path, landmarks)  # BGR
    return img


def transform(img, flip=False):
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


def get_feature(model, samples, file):
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
    img = get_image(samples, file)
    imgs[0] = transform(img.copy(), False)
    imgs[1] = transform(img.copy(), True)
    with torch.no_grad():
        output = model(imgs)
    feature_0 = output[0].cpu().numpy()
    feature_1 = output[1].cpu().numpy()
    feature = feature_0 + feature_1
    return feature / np.linalg.norm(feature)


def evaluate(model):
    model.eval()

    with open(custom_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    filename = 'data/custom_test_pair.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()

    angles = []

    elapsed = 0

    for line in tqdm(lines):
        tokens = line.split()

        start = time.time()
        x0 = get_feature(model, samples, tokens[0])
        x1 = get_feature(model, samples, tokens[1])

        end = time.time()
        elapsed += end - start
        # pdb.set_trace()
        cosine = np.dot(x0, x1)
        cosine = np.clip(cosine, -1.0, 1.0)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi
        # print(theta)
        is_same = tokens[2]
        angles.append('{} {}\n'.format(theta, is_same))

    print('elapsed: {} ms'.format(elapsed / (6000 * 2) * 1000))

    with open('data/custom_angles.txt', 'w') as file:
        file.writelines(angles)


def visualize(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    ones = []
    zeros = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            ones.append(angle)
        else:
            zeros.append(angle)

    bins = np.linspace(0, 180, 181)

    plt.hist(zeros, bins, density=True, alpha=0.5, label='0', facecolor='red')
    plt.hist(ones, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu_0 = np.mean(zeros)
    sigma_0 = np.std(zeros)
    y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
    plt.plot(bins, y_0, 'r--')
    mu_1 = np.mean(ones)
    sigma_1 = np.std(ones)
    y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
    plt.plot(bins, y_1, 'b--')
    plt.xlabel('theta')
    plt.ylabel('theta j Distribution')
    plt.title(
        r'Histogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.format(mu_0, sigma_0, mu_1, sigma_1))

    print('threshold: ' + str(threshold))
    print('mu_0: ' + str(mu_0))
    print('sigma_0: ' + str(sigma_0))
    print('mu_1: ' + str(mu_1))
    print('sigma_1: ' + str(sigma_1))

    plt.legend(loc='upper right')
    plt.plot([threshold, threshold], [0, 0.05], 'k-', lw=2)
    ensure_folder('images')
    plt.savefig('images/theta_dist.png')
    # plt.show()


def accuracy(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    wrong = 0
    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            if angle > threshold:
                wrong += 1
        else:
            if angle <= threshold:
                wrong += 1

    accuracy = 1 - wrong / 138
    return accuracy


def show_bboxes(folder):
    with open(custom_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']
    for sample in tqdm(samples):
        full_path = sample['full_path']
        bounding_boxes = sample['bounding_boxes']
        landmarks = sample['landmarks']
        img = cv.imread(full_path)
        img = draw_bboxes(img, bounding_boxes, landmarks)
        filename = os.path.basename(full_path)
        filename = os.path.join(folder, filename)
        cv.imwrite(filename, img)


def error_analysis(threshold):
    with open(angles_file) as file:
        angle_lines = file.readlines()

    fp = []
    fn = []
    for i, line in enumerate(angle_lines):
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if angle <= threshold and type == 0:
            fp.append(i)
        if angle > threshold and type == 1:
            fn.append(i)

    print('len(fp): ' + str(len(fp)))
    print('len(fn): ' + str(len(fn)))

    num_fp = len(fp)
    num_fn = len(fn)

    filename = 'data/custom_test_pair.txt'
    with open(filename, 'r') as file:
        pair_lines = file.readlines()

    for i in range(num_fp):
        fp_id = fp[i]
        fp_line = pair_lines[fp_id]
        tokens = fp_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fp_0.jpg'.format(i))
        save_aligned(file0, '{}_fp_0_aligned.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fp_1.jpg'.format(i))
        save_aligned(file1, '{}_fp_1_aligned.jpg'.format(i))

    for i in range(num_fn):
        fn_id = fn[i]
        fn_line = pair_lines[fn_id]
        tokens = fn_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fn_0.jpg'.format(i))
        save_aligned(file0, '{}_fn_0_aligned.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fn_1.jpg'.format(i))
        save_aligned(file1, '{}_fn_1_aligned.jpg'.format(i))


def save_aligned(old_fn, new_fn):
    # old_fn = os.path.join( "/home/ahmadob/dataset/facerecognition_dataset/test_set", old_fn)
    is_valid, bounding_boxes, landmarks = get_central_face_attributes(old_fn)
    img = align_face(old_fn, landmarks)
    new_fn = os.path.join('images', new_fn)
    cv.imwrite(new_fn, img)


def copy_file(old_fn, new):
    # old_fn = os.path.join("/home/ahmadob/dataset/facerecognition_dataset/test_set", old)
    img = cv.imread(old_fn)
    bounding_boxes, landmarks = get_all_face_attributes(old_fn)
    draw_bboxes(img, bounding_boxes, landmarks)
    cv.resize(img, (224, 224))
    new_fn = os.path.join('images', new)
    cv.imwrite(new_fn, img)


def get_threshold():
    with open(angles_file, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'angle': angle, 'type': type})

    min_error = 138
    min_threshold = 0

    for d in data:
        threshold = d['angle']
        type1 = len([s for s in data if s['angle'] <= threshold and s['type'] == 0])
        type2 = len([s for s in data if s['angle'] > threshold and s['type'] == 1])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    # print(min_error, min_threshold)
    return min_threshold


def custom_test(model):
    filename = '/home/ahmadob/dataset/facerecognition_dataset/test_set'

    # if not os.path.isfile(lfw_pickle):
    print('Processing {}...'.format(custom_pickle))
    process()

    # if not os.path.isfile(angles_file):
    print('Evaluating {}...'.format(angles_file))
    evaluate(model)

    print('Calculating threshold...')
    # threshold = 70.36
    thres = get_threshold()
    print('Calculating accuracy...')
    acc = accuracy(thres)
    print('Accuracy: {}%, threshold: {}'.format(acc * 100, thres))
    return acc, thres


if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()

    # scripted_model_file = 'pretrained_model/mobilefacenet_scripted.pt'
    # model = torch.jit.load(scripted_model_file)
    # model = model.to(device)
    # model.eval()

    acc, threshold = custom_test(model)

    print('Visualizing {}...'.format(angles_file))
    visualize(threshold)

    print('error analysis...')
    error_analysis(threshold)
