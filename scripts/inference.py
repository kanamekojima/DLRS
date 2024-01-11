from argparse import ArgumentParser
import os
import pathlib
import sys

import cv2
import numpy as np

import deeplabv3


COLOR_DICT = {
    'nucleus': (0, 255, 0),
    'glomerulus': (0, 180, 255),
    'interstitium': (255, 160, 0),
    'tubule': (255, 0, 180),
    'artery': (0, 0, 255),
}
PATCH_SIZE = 512


def mkdir(dirname):
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


def get_point_list(size, patch_size, stride=None):
    point_list = []
    break_flag = False
    point = 0
    while break_flag is False:
        if point + patch_size >= size:
            if stride is not None:
                point = size - patch_size
            break_flag = True
        if point >= 0:
            point_list.append(point)
        if stride is not None:
            point += stride
        else:
            point += patch_size
    return point_list


def get_top_left_point_list(width, height, patch_size, stride):
    y_list = get_point_list(height, patch_size, stride)
    x_list = get_point_list(width, patch_size, stride)
    top_left_point_list = []
    for y in y_list:
        for x in x_list:
            top_left_point_list.append([x, y])
    return top_left_point_list


def get_annotated_image(image, label, label_name_list, alpha=0.5):
    binary_map = np.zeros(label.shape, np.uint8)
    annotation = np.zeros([*label.shape, 3], np.uint8)
    for i, label_name in enumerate(label_name_list, start=1):
        binary_map[label == i] = 1
        annotation[label == i] = COLOR_DICT[label_name]
    index_set = np.where(binary_map == 1)
    annotated_image = image.copy()
    annotated_image[index_set] = np.clip(
        alpha * annotation[index_set] +
        (1.0 - alpha) * image[index_set], 0, 255).astype(np.uint8)
    return annotated_image


def main():
    description = 'inference'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--image-file', type=str, required=True,
                        dest='image_file', help='image file')
    parser.add_argument('--output-file', type=str, required=True,
                        dest='output_file', help='output file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        dest='checkpoint_file', help='checkpoint file')
    parser.add_argument('--batch-size', type=int, required=True,
                        dest='batch_size', help='batch size')
    parser.add_argument('--patch-buffer-size', type=int, default=5000,
                        dest='patch_buffer_size', help='patch buffer size')
    parser.add_argument('--patch-stride', type=int, default=128,
                        dest='patch_stride', help='patch stride')
    parser.add_argument('--segmentation-type', type=str, required=True,
                        dest='segmentation_type',
                        help='segmentation type [tissue / nucleus]')
    args = parser.parse_args()

    assert args.segmentation_type in {'tissue', 'nucleus'}
    if args.segmentation_type == 'tissue':
        label_name_list = ['interstitium', 'tubule', 'glomerulus', 'artery']
    elif args.segmentation_type == 'nucleus':
        label_name_list = ['nucleus']
    else:
        print(
            'Unsupported segmentation type: ' + args.segmentation_type,
            file=sys.stderr
        )
        sys.exit(0)
    num_classes = len(label_name_list) + 1
    image = cv2.imread(args.image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prob_map = np.zeros([*image.shape[:2], num_classes], np.float32)
    count_map = np.zeros(image.shape[:2], np.int32)
    decoder = deeplabv3.Decoder(args.checkpoint_file, num_classes)
    height, width = image.shape[:2]
    top_left_point_list = get_top_left_point_list(
        width, height, PATCH_SIZE, args.patch_stride)
    num_points = len(top_left_point_list)
    index = 0
    while index < num_points:
        next_index = min(index + args.patch_buffer_size, num_points)
        top_left_point_sublist = top_left_point_list[index: next_index]
        index = next_index
        image_patch_list = [
            image[y: y + PATCH_SIZE, x: x + PATCH_SIZE]
            for x, y in top_left_point_sublist
        ]
        prob_map_patch_list = decoder.decode(
            image_patch_list, args.batch_size)
        for i, prob_map_patch in enumerate(prob_map_patch_list):
            x, y = top_left_point_sublist[i]
            prob_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE] += prob_map_patch
            count_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE] += 1
    zero_count_map = (count_map == 0).astype(count_map.dtype)
    prob_map /= np.expand_dims(count_map + zero_count_map, axis=-1)
    label = np.argmax(prob_map, axis=-1).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_image = get_annotated_image(
        image, label, label_name_list)
    mkdir(os.path.dirname(args.output_file))
    cv2.imwrite(args.output_file, annotated_image)


if __name__ == '__main__':
    main()
