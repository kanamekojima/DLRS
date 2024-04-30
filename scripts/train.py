from argparse import ArgumentParser
import os
import sys

import cv2
import numpy as np

import deeplabv3
import data_loader


def get_label_weights(label_frequencies, rate=0.5):
    """
    Compute label weights based on the class frequencies. The weights are adjusted
    by the specified rate to avoid overfitting due to imbalanced classes.
    """
    # Compute initial weights based on inverse frequency
    label_weights = [
        pow(1.0 / max(label_frequency, 1e-5), rate)
        for label_frequency in label_frequencies
    ]

    # Normalize weights so that they sum to the number of classes
    norm = sum(label_weights)
    num_classes = len(label_weights)
    label_weights = [
        label_weight * num_classes / norm for label_weight in label_weights
    ]
    return label_weights


def main():
    description = 'train'
    parser = ArgumentParser(description=description, add_help=False)
    parser.add_argument('--train-list', type=str, required=True,
                        dest='train_list_file', help='train file list')
    parser.add_argument('--validation-list', type=str, required=True,
                        dest='validation_list_file',
                        help='valiadtion list file')
    parser.add_argument('--output-dir', type=str, required=True,
                        dest='output_dir', help='output dir')
    parser.add_argument('--output-basename', type=str, required=True,
                        dest='output_basename', help='output basename')
    parser.add_argument('--num-classes', type=int, required=True,
                        dest='num_classes', help='No. of classes')
    parser.add_argument('--train-batch-size', type=int, required=True,
                        dest='train_batch_size', help='train batch size')
    parser.add_argument('--validation-batch-size', type=int, required=True,
                        dest='validation_batch_size',
                        help='validation batch size')
    parser.add_argument('--patch-size', type=int, default=512,
                        dest='patch_size', help='patch size')
    parser.add_argument('--patch-margin', type=int, default=38,
                        dest='patch_margin', help='patch margin')
    parser.add_argument('--patch-stride', type=int, default=20,
                        dest='patch_stride', help='patch stride')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        dest='learning_rate', help='learning rate')
    parser.add_argument('--segmentation-type', type=str, required=True,
                        dest='segmentation_type',
                        help='segmentation type [tissue / nucleus]')
    parser.add_argument('--iteration-count', type=int, default=25000,
                        dest='iteration_count', help='iteration count')
    parser.add_argument('--seed', type=int, default=3141592653,
                        dest='random_seed', help='random seed')
    args = parser.parse_args()

    # Set the random seed for reproducibility
    np.random.seed(args.random_seed)

    # Read training data from file
    file_dict_list = []
    with open(args.train_list_file, 'rt') as fin:
        for line in fin:
            image_file, label_file = line.rstrip().split()
            file_dict_list.append({
                'image file': image_file,
                'label file': label_file,
            })

    # Prepare training data
    train_data_prefix = os.path.join(
        args.output_dir, 'train_data', args.output_basename)
    data_loader_config = data_loader.Config(
        args.num_classes,
        args.patch_size + args.patch_margin,
        args.patch_stride,
    )
    data_loader.prepare_train_data(
        data_loader_config, file_dict_list, train_data_prefix)
    train_data_loader, class_frequency = data_loader.load_train_data(
        train_data_prefix + '.dat')

    # Prepare validation data
    validation_image_patches = []
    validation_label_patches = []
    data_loader_config.patch_size = args.patch_size
    data_loader_config.stride = args.patch_size // 2
    with open(args.validation_list_file, 'rt') as fin:
        for line in fin:
            image_file, label_file = line.rstrip().split()
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)  # Load label
            image_patch_list, label_patch_list = data_loader.load_patch_data(
                data_loader_config, image, label)
            validation_image_patches.extend(image_patch_list)
            validation_label_patches.extend(label_patch_list)
    validation_image_patches = np.array(validation_image_patches, np.uint8)
    validation_label_patches = np.array(validation_label_patches, np.uint8)
    print('Validation patch count: {:d}'.format(len(validation_image_patches)))

    # Normalize class frequencies and compute label weights
    class_frequency /= np.sum(class_frequency)
    class_frequency = class_frequency.tolist()
    print('Class frequency: ' + ', '.join(map(str, class_frequency)))
    label_weights = get_label_weights(class_frequency, 0.5)

    # Configure deep learning model for training
    assert args.segmentation_type in {'tissue', 'nucleus'}
    config = deeplabv3.Config(
        patch_shape=[args.patch_size, args.patch_size],
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        segmentation_type=args.segmentation_type,
        iteration_count=args.iteration_count,
    )

    # Train the deep learning model
    checkpoint_prefix = os.path.join(
        args.output_dir, 'checkpoints', args.output_basename)

    deeplabv3.train(
        config,
        train_data_loader,
        label_weights,
        validation_image_patches,
        validation_label_patches,
        checkpoint_prefix,
        random_seed=args.random_seed,
    )


if __name__ == '__main__':
    main()
