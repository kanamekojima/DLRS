from argparse import ArgumentParser
import os
import pathlib
import sys

import cv2
import numpy as np

import deeplabv3


# Color dictionary for annotations
COLOR_DICT = {
    'nucleus': (0, 255, 0),
    'glomerulus': (0, 180, 255),
    'interstitium': (255, 160, 0),
    'tubule': (255, 0, 180),
    'artery': (0, 0, 255),
}
# Default patch size for segmentation
PATCH_SIZE = 512


def mkdir(dirname):
    """Create a directory if it doesn't exist"""
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


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


def get_annotated_image(image, label, label_name_list, alpha=0.5):
    """Overlay segmentation annotations on the image"""
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

    # Set label names based on segmentation type
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

    # Load the input image and convert to RGB
    image = cv2.imread(args.image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize probability map and count map for averaging overlapping patches
    prob_map = np.zeros([*image.shape[:2], num_classes], np.float32)
    count_map = np.zeros(image.shape[:2], np.int32)

    # Initialize the deep learning model for decoding/segmentation
    decoder = deeplabv3.Decoder(args.checkpoint_file, num_classes)

    # Get image dimensions and generate list of top-left points for patches
    height, width = image.shape[:2]
    top_left_point_list = get_top_left_point_list(
        width, height, PATCH_SIZE, args.patch_stride
    )
    num_points = len(top_left_point_list)

    # Perform sliding window inference on the image
    index = 0
    while index < num_points:
        # Process patches in batches to save memory
        next_index = min(index + args.patch_buffer_size, num_points)
        top_left_point_sublist = top_left_point_list[index: next_index]
        index = next_index

        # Extract patches from the slide image
        image_patch_list = [
            image[y: y + PATCH_SIZE, x: x + PATCH_SIZE]
            for x, y in top_left_point_sublist
        ]

        # Decode the patches to get probability maps
        prob_map_patch_list = decoder.decode(image_patch_list, args.batch_size)

        # Update the probability map with decoded patches
        for i, prob_map_patch in enumerate(prob_map_patch_list):
            x, y = top_left_point_sublist[i]
            prob_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE] += prob_map_patch
            count_map[y: y + PATCH_SIZE, x: x + PATCH_SIZE] += 1

    # Avoid division by zero and normalize the probability map
    zero_count_map = (count_map == 0).astype(count_map.dtype)
    prob_map /= np.expand_dims(count_map + zero_count_map, axis=-1)

    # Determine class labels based on highest probability
    label = np.argmax(prob_map, axis=-1).astype(np.uint8)

    # Convert the image back to BGR format for OpenCV compatibility
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get the annotated image with segmentation results
    annotated_image = get_annotated_image(image, label, label_name_list)

    # Save the annotated image to the output file
    mkdir(os.path.dirname(args.output_file))
    cv2.imwrite(args.output_file, annotated_image)


if __name__ == '__main__':
    main()
