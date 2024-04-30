import os
import pathlib
import sys

import cv2
import imutils
import numpy as np


# List of rotation angles to be applied to data augmentation
ROTATION_ANGLE_LIST = [None, 15, 30, 45, 60, 75]


def mkdir(dirname):
    """Create a directory if it doesn't exist"""
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


class DataLoader:
    """
    DataLoader for managing the loading and preparation of training data
    with specified patch size and angle variations.
    """

    def __init__(self, npz_file_list, image_patch_count_list, patch_size):
        # List of preprocessed data files
        self.npz_file_list = npz_file_list
        # Number of image patches in each data file
        self.image_patch_count_list = image_patch_count_list
        # Size of the image patches to be extracted
        self.patch_size = patch_size
        # Total number of image patches across all data files
        self.total_image_patch_count = sum(self.image_patch_count_list)
        # Number of different rotation angles
        self.num_angle_types = len(ROTATION_ANGLE_LIST)
        # Number of unique images used to generate patches
        self.num_actual_images = len(npz_file_list) // self.num_angle_types

    def get_sampled_count_list(self, image_patch_count_limit):
        """
        Generate a list of sampled counts of image patches based on a limit.
        This ensures a balanced distribution of patch samples for training.
        """
        total_image_patch_count = 0
        image_file_index_list = []
        # Randomly select rotation angles for each image
        for i in range(self.num_actual_images):
            rs = np.random.choice(
                self.num_angle_types, len(ROTATION_ANGLE_LIST), replace=False
            )
            image_file_index_list.append(rs)
            for r in rs:
                total_image_patch_count += self.image_patch_count_list[
                    self.num_angle_types * i + r
                ]

        # Calculate the proportion of sampling to achieve the limit
        loading_rate = image_patch_count_limit / total_image_patch_count
        remaining_count = image_patch_count_limit
        sampled_patch_count_list = [0] * len(self.npz_file_list)

        # Distribute sampled counts based on the calculated loading rate
        for i in range(self.num_actual_images):
            for image_file_index in image_file_index_list[i]:
                j = self.num_angle_types * i + image_file_index
                image_patch_count = self.image_patch_count_list[j]
                sampled_patch_count = int(image_patch_count * loading_rate)
                sampled_patch_count_list[j] = sampled_patch_count
                remaining_count -= sampled_patch_count

        # Adjust to ensure the total count meets the limit
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
        """
        Shuffle two lists while maintaining their paired order.
        """
        list_pair = list(zip(list1, list2))
        np.random.shuffle(list_pair)
        return zip(*list_pair)

    def load_data(self, image_patch_count_limit=10000):
        """
        Load training data with a specified limit on the number of patches.
        This function samples and shuffles the image patches for training.
        """
        sampled_image_patch_count_list = self.get_sampled_count_list(
            image_patch_count_limit)
        image_patches = []
        label_patches = []
        image_patch_count = 0

        # Load sampled patches from preprocessed data files
        for i, npz_file in enumerate(self.npz_file_list):
            m = sampled_image_patch_count_list[i]
            if m == 0:
                continue  # Skip if no patches are sampled from this file
            n = self.image_patch_count_list[i]
            npz = np.load(npz_file)
            image = npz['image']
            top_left_point_list = npz['top_left_point_list']  # List of top-left points for patches

            # Select random patches based on the sampled count
            choice = np.random.choice(n, m, replace=False)
            for x, y in top_left_point_list[choice]:
                # Extract and append patches to the training set
                patch = np.copy(
                    image[y : y + self.patch_size, x : x + self.patch_size]
                )
                image_patches.append(patch[:, :, :3])
                label_patches.append(patch[:, :, 3])
                image_patch_count += 1
            del image, npz, top_left_point_list

        # Shuffle the patches to ensure randomness
        image_patches, label_patches = self.shuffle_list_pair(
            image_patches, label_patches)
        image_patches = np.array(image_patches, np.uint8)
        label_patches = np.array(label_patches, np.uint8)
        return image_patches, label_patches


class Config:
    """
    Configuration class to define patch size, stride, and number of classes
    for loading and processing data.
    """

    def __init__(self, num_classes, patch_size, stride):
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.stride = stride


def get_point_list(size, patch_size, stride):
    """
    Generate a list of top-left starting points for patch extraction
    given the total size, patch size, and stride.
    """
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
    """
    Create a list of top-left points for patch extraction
    given the width, height, patch size, and stride.
    """
    y_list = get_point_list(height, patch_size, stride)  # Vertical points
    x_list = get_point_list(width, patch_size, stride)  # Horizontal points
    top_left_point_list = []
    # Create pairs of (x, y) coordinates for patch extraction
    for y in y_list:
        for x in x_list:
            top_left_point_list.append([x, y])
    return top_left_point_list


def get_rotated_binary_map(binary_map, rotation_angle):
    """
    Rotate a binary map (like a label map) by the specified angle.
    This is useful for data augmentation during training.
    """
    height, width = binary_map.shape[:2]
    # Rotate the binary map and threshold to obtain a clear binary structure
    binary_map = imutils.rotate_bound(binary_map * 255, angle=rotation_angle)
    _, binary_map = cv2.threshold(binary_map, 127, 1, cv2.THRESH_BINARY)
    return binary_map


def get_rotated_images(image, one_hot_label, rotation_angle):
    """
    Rotate an image and its corresponding one-hot encoded labels
    by the specified rotation angle.
    This is used for data augmentation to create new variations.
    """
    height, width = image.shape[:2]
    # Rotate the image
    image = imutils.rotate_bound(image, angle=rotation_angle).astype(np.uint8)
    # Rotate each one-hot encoded label
    one_hot_label = np.array([
        get_rotated_binary_map(binary_map, rotation_angle)
        for binary_map in one_hot_label
    ])
    # Create a non-image map to represent rotated empty space
    image_map = imutils.rotate_bound(
        np.ones([height, width], dtype=np.uint8), angle=rotation_angle
    )
    non_image_map = (1 - image_map).astype(np.int32)  # Mark areas with no data
    return image, one_hot_label, non_image_map


def get_patch_image_info(config, image, label, rotation_angle=None):
    """
    Extract patches from an image and its corresponding label.
    Supports rotation if specified by the rotation angle.
    """
    if rotation_angle is not None and rotation_angle % 360 != 0:
        one_hot_label = np.array([
            (label == i).astype(np.uint8) for i in range(config.num_classes)
        ])
        image, one_hot_label, non_image_map = get_rotated_images(
            image, one_hot_label, rotation_angle
        )
        # Reconstruct the label after rotation
        label = np.zeros(image.shape[:2], dtype=np.uint8)
        for i, binary_map in enumerate(one_hot_label):
            label[binary_map == 1] = i
    else:
        non_image_map = np.zeros(image.shape[:2], dtype=np.int32)

    height, width = image.shape[:2]
    patch_size = config.patch_size
    top_left_point_list = get_top_left_point_list(
        width, height, patch_size, config.stride
    )

    # Select only valid patches that do not contain non-image data
    selected_top_left_point_list = []
    for x, y in top_left_point_list:
        non_image_pixel_count = np.sum(
            non_image_map[y: y + patch_size, x: x + patch_size]
        )
        if non_image_pixel_count > 0:
            continue
        selected_top_left_point_list.append((x, y))

    return {
        'image': image,
        'label': label,
        'top_left_point_list': selected_top_left_point_list,
    }


def prepare_train_data(config, file_dict_list, output_prefix):
    """Prepare training data with data augmentation"""
    mkdir(os.path.dirname(output_prefix))

    npz_file_list = []  # List of .npz files containing augmented data
    image_patch_count_list = []  # List of patch counts for each file
    class_frequency = np.zeros(config.num_classes, np.float32)  # Class distribution
    num_files = len(file_dict_list)

    # Iterate over all input files to prepare training data
    for i, file_dict in enumerate(file_dict_list, start=1):
        image_file = file_dict['image file']
        label_file = file_dict['label file']
        print('{:d} / {:d} Loading {:s} ...'.format(
            i, num_files, image_file), flush=True
        )

        # Read image and convert color for processing
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read label in grayscale mode for segmentation
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        # Generate patches with different rotation angles for augmentation
        for rotation_angle in ROTATION_ANGLE_LIST:
            patch_image_info = get_patch_image_info(
                config, image, label, rotation_angle
            )
            top_left_point_list = patch_image_info['top_left_point_list']
            image_patch_count = len(top_left_point_list)  # Number of patches

            if image_patch_count == 0:
                continue  # Skip if no patches are extracted

            # Generate unique output file name based on rotation angle
            stem = pathlib.Path(image_file).stem
            if rotation_angle is not None:
                npz_file = '{:s}_{:s}_angle_{:d}.npz'.format(
                    output_prefix, stem, rotation_angle
                )
            else:
                npz_file = '{:s}_{:s}.npz'.format(output_prefix, stem)

            # Save the image along with the label and top-left points
            image_to_save = [
                patch_image_info['image'],
                np.expand_dims(patch_image_info['label'], -1),
            ]
            image_to_save = np.concatenate(image_to_save, axis=-1)
            np.savez(
                os.path.splitext(npz_file)[0],
                image=image_to_save,
                top_left_point_list=np.array(top_left_point_list, np.int32)
            )

            npz_file_list.append(npz_file)
            image_patch_count_list.append(image_patch_count)

            # Update class frequency only for unrotated patches
            if rotation_angle is None:
                extraction_map = np.zeros(image.shape[:2], np.int32)
                patch_size = config.patch_size
                for x, y in top_left_point_list:
                    extraction_map[y: y + patch_size, x: x + patch_size] = 1
                for i in range(config.num_classes):
                    class_frequency[i] += np.sum(
                        (label == i).astype(np.int32) * extraction_map
                    )

    # Save class frequencies for later use
    np.savez(output_prefix + '_freq', class_frequency)

    # Save the training data to a .dat file
    with open(output_prefix + '.dat', 'wt') as fout:
        fout.write(output_prefix + '_freq.npz\n')  # Class frequency data
        fout.write('{:d}\n'.format(config.patch_size))  # Patch size

        # Write patch counts and corresponding file names
        for image_patch_count, npz_file in zip(
                image_patch_count_list, npz_file_list
        ):
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
            label[y: y + patch_size, x: x + patch_size]
        )
    return image_patch_list, label_patch_list


def load_train_data(dat_file):
    """
    Load training data from the specified .dat file,
    which contains the list of .npz files with training data patches.
    """
    npz_file_list = []
    class_frequency = None
    total_image_patch_count = 0
    image_patch_count_list = []
    with open(dat_file, 'rt') as fin:
        npz_file = fin.readline().rstrip()  # Load class frequency file
        npz = np.load(npz_file)
        class_frequency = npz['arr_0']
        patch_size = int(fin.readline().rstrip())

        # Load all .npz files and patch counts from the data file
        for line in fin:
            image_patch_count, npz_file = line.rstrip().split('\t')
            image_patch_count = int(image_patch_count)
            image_patch_count_list.append(image_patch_count)
            npz_file_list.append(npz_file)

    # Initialize the DataLoader with the loaded information
    data_loader = DataLoader(npz_file_list, image_patch_count_list, patch_size)
    return data_loader, class_frequency
