import os
import pathlib
import sys

import cv2
import imutils
import numpy as np


ROTATION_ANGLE_LIST = [None, 15, 30, 45, 60, 75]


def mkdir(dirname):
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


class DataLoader:

    def __init__(self, npz_file_list, image_patch_count_list, patch_size):
        self.npz_file_list = npz_file_list
        self.image_patch_count_list = image_patch_count_list
        self.patch_size = patch_size
        self.total_image_patch_count = sum(self.image_patch_count_list)
        self.num_angle_types = len(ROTATION_ANGLE_LIST)
        self.num_actual_images = len(npz_file_list) // self.num_angle_types

    def get_sampled_count_list(self, image_patch_count_limit):
        total_image_patch_count = 0
        image_file_index_list = []
        for i in range(self.num_actual_images):
            rs = np.random.choice(
                self.num_angle_types, len(ROTATION_ANGLE_LIST), replace=False)
            image_file_index_list.append(rs)
            for r in rs:
                total_image_patch_count += self.image_patch_count_list[
                    self.num_angle_types * i + r]
        loading_rate = image_patch_count_limit / total_image_patch_count
        remaining_count = image_patch_count_limit
        sampled_patch_count_list = [0] * len(self.npz_file_list)
        for i in range(self.num_actual_images):
            for image_file_index in image_file_index_list[i]:
                j = self.num_angle_types * i + image_file_index
                image_patch_count = self.image_patch_count_list[j]
                sampled_patch_count = int(image_patch_count * loading_rate)
                sampled_patch_count_list[j] = sampled_patch_count
                remaining_count -= sampled_patch_count
        while remaining_count > 0:
            m = min(remaining_count, self.num_actual_images)
            choice = np.random.choice(self.num_actual_images, m, replace=False)
            for i in choice:
                for image_file_index in image_file_index_list[i]:
                    j = self.num_angle_types * i + image_file_index
                    if sampled_patch_count_list[j] < \
                            self.image_patch_count_list[j]:
                        sampled_patch_count_list[j] += 1
                        remaining_count -= 1
        return sampled_patch_count_list

    def shuffle_list_pair(self, list1, list2):
        list_pair = list(zip(list1, list2))
        np.random.shuffle(list_pair)
        return zip(*list_pair)

    def load_data(self, image_patch_count_limit=10000):
        sampled_image_patch_count_list = self.get_sampled_count_list(
            image_patch_count_limit)
        image_patches = []
        label_patches = []
        image_patch_count = 0
        for i, npz_file in enumerate(self.npz_file_list):
            m = sampled_image_patch_count_list[i]
            if m == 0:
                continue
            n = self.image_patch_count_list[i]
            npz = np.load(npz_file)
            image = npz['image']
            top_left_point_list = npz['top_left_point_list']
            choice = np.random.choice(n, m, replace=False)
            for x, y in top_left_point_list[choice]:
                patch = np.copy(image[
                    y : y + self.patch_size, x : x + self.patch_size])
                image_patches.append(patch[:, :, :3])
                label_patches.append(patch[:, :, 3])
                image_patch_count += 1
            del image, npz, top_left_point_list
        image_patches, label_patches = self.shuffle_list_pair(
            image_patches, label_patches)
        image_patches = np.array(image_patches, np.uint8)
        label_patches = np.array(label_patches, np.uint8)
        return image_patches, label_patches


class Config:

    def __init__(self, num_classes, patch_size, stride):
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.stride = stride


def get_point_list(size, patch_size, stride):
    point_list = []
    break_flag = False
    point = 0
    while break_flag is False:
        if point + patch_size >= size:
            point = size - patch_size
            break_flag = True
        if point >= 0:
            point_list.append(point)
        point += stride
    return point_list


def get_top_left_point_list(width, height, patch_size, stride):
    y_list = get_point_list(height, patch_size, stride)
    x_list = get_point_list(width, patch_size, stride)
    top_left_point_list = []
    for y in y_list:
        for x in x_list:
            top_left_point_list.append([x, y])
    return top_left_point_list


def get_rotated_binary_map(binary_map, rotation_angle):
    height, width = binary_map.shape[:2]
    binary_map = imutils.rotate_bound(binary_map * 255, angle=rotation_angle)
    _, binary_map = cv2.threshold(binary_map, 127, 1, cv2.THRESH_BINARY)
    return binary_map


def get_rotated_images(image, one_hot_label, rotation_angle):
    height, width = image.shape[:2]
    image = imutils.rotate_bound(image, angle=rotation_angle).astype(np.uint8)
    one_hot_label = np.array([
        get_rotated_binary_map(binary_map, rotation_angle)
        for binary_map in one_hot_label
    ])
    image_map = imutils.rotate_bound(
        np.ones([height, width], dtype=np.uint8), angle=rotation_angle)
    non_image_map = (1 - image_map).astype(np.int32)
    return image, one_hot_label, non_image_map


def get_patch_image_info(
        config,
        image,
        label,
        rotation_angle=None,
        ):
    if rotation_angle is not None and rotation_angle % 360 != 0:
        one_hot_label = np.array([
            (label == i).astype(np.uint8) for i in range(config.num_classes)
        ])
        image, one_hot_label, non_image_map = get_rotated_images(
            image, one_hot_label, rotation_angle)
        label = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, binary_map in enumerate(one_hot_label):
            label[binary_map == 1] = i
    else:
        non_image_map = np.zeros(image.shape[:2], dtype=np.int32)
    height, width = image.shape[:2]
    patch_size = config.patch_size
    top_left_point_list = get_top_left_point_list(
        width, height, patch_size, config.stride)
    selected_top_left_point_list = []
    for x, y in top_left_point_list:
        non_image_pixel_count = np.sum(
            non_image_map[y: y + patch_size, x: x + patch_size])
        if non_image_pixel_count > 0:
            continue
        selected_top_left_point_list.append((x, y))
    return {
        'image': image,
        'label': label,
        'top_left_point_list': selected_top_left_point_list,
    }


def prepare_train_data(config, file_dict_list, output_prefix):
    mkdir(os.path.dirname(output_prefix))
    npz_file_list = []
    image_patch_count_list = []
    class_frequency = np.zeros(config.num_classes, np.float32)
    num_files = len(file_dict_list)
    for i, file_dict in enumerate(file_dict_list, start=1):
        image_file = file_dict['image file']
        label_file = file_dict['label file']
        print('{:d} / {:d} Loading {:s} ...'.format(
            i, num_files, image_file), flush=True)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        for rotation_angle in ROTATION_ANGLE_LIST:
            patch_image_info = get_patch_image_info(
                config, image, label, rotation_angle)
            top_left_point_list = patch_image_info['top_left_point_list']
            image_patch_count = len(top_left_point_list)
            if image_patch_count == 0:
                continue
            stem = pathlib.Path(image_file).stem
            if rotation_angle is not None:
                npz_file = '{:s}_{:s}_angle_{:d}.npz'.format(
                    output_prefix, stem, rotation_angle)
            else:
                npz_file = '{:s}_{:s}.npz'.format(output_prefix, stem)
            image_to_save = [
                patch_image_info['image'],
                np.expand_dims(patch_image_info['label'], -1),
            ]
            image_to_save = np.concatenate(image_to_save, axis=-1)
            np.savez(
                os.path.splitext(npz_file)[0],
                image=image_to_save,
                top_left_point_list=np.array(top_left_point_list, np.int32))
            npz_file_list.append(npz_file)
            image_patch_count_list.append(image_patch_count)
            if rotation_angle is None:
                extraction_map = np.zeros(image.shape[:2], np.int32)
                patch_size = config.patch_size
                for x, y in top_left_point_list:
                    extraction_map[y: y + patch_size, x: x + patch_size] = 1
                for i in range(config.num_classes):
                    class_frequency[i] += np.sum(
                        (label == i).astype(np.int32) * extraction_map)

    np.savez(output_prefix + '_freq', class_frequency)
    with open(output_prefix + '.dat', 'wt') as fout:
        fout.write(output_prefix + '_freq.npz\n')
        fout.write('{:d}\n'.format(config.patch_size))
        for image_patch_count, npz_file in zip(
                image_patch_count_list, npz_file_list):
            fout.write('{:d}\t{:s}\n'.format(image_patch_count, npz_file))


def load_patch_data(config, image, label):
    patch_image_info = get_patch_image_info(
        config, image, label, rotation_angle=None)
    label = patch_image_info['label']
    top_left_point_list = patch_image_info['top_left_point_list']
    image_patch_list = []
    label_patch_list = []
    patch_size = config.patch_size
    for x, y in top_left_point_list:
        image_patch_list.append(image[y: y + patch_size, x: x + patch_size])
        label_patch_list.append(
            label[y: y + patch_size, x: x + patch_size])
    return image_patch_list, label_patch_list


def load_train_data(dat_file):
    npz_file_list = []
    class_frequency = None
    total_image_patch_count = 0
    image_patch_count_list = []
    with open(dat_file, 'rt') as fin:
        npz_file = fin.readline().rstrip()
        npz = np.load(npz_file)
        class_frequency = npz['arr_0']
        patch_size = int(fin.readline().rstrip())
        for line in fin:
            image_patch_count, npz_file = line.rstrip().split('\t')
            image_patch_count = int(image_patch_count)
            image_patch_count_list.append(image_patch_count)
            npz_file_list.append(npz_file)

    data_loader = DataLoader(npz_file_list, image_patch_count_list, patch_size)
    return data_loader, class_frequency
