import numpy as np
import torch
from torchvision import transforms


# Mean and standard deviation for normalization
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


class TrainDataset(torch.utils.data.Dataset):
    """
    Dataset class for training, supporting augmentation and normalization.
    """

    def __init__(
            self,
            images,
            labels,
            crop_shape=None,
            apply_augmentation=True,
    ):
        # Initialize image and label data
        self.images = np.transpose(images, axes=(0, 3, 1, 2))
        self.images = torch.from_numpy(self.images)
        self.labels = np.expand_dims(labels, axis=1)
        self.labels = torch.from_numpy(self.labels)
        self.datanum = self.images.size(0)

        self.apply_augmentation = apply_augmentation  # Whether to apply augmentation

        if self.apply_augmentation:
            transform_list = []  # List of transformation operations

            # Apply random cropping if specified
            if crop_shape is not None:
                transform_list.append(transforms.RandomCrop(crop_shape))

            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply(
                    [lambda x: torch.rot90(x, dims=[1, 2])]  # Random 90-degree rotation
                ),
            ])

            # Compose the transformations for shape and color
            self.shape_transform = transforms.Compose(transform_list)
            self.color_transform = transforms.ColorJitter(
                brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2
            )

        # Normalize the images with standard mean and standard deviation
        self.normalize = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32) / 255),  # Normalize to [0, 1]
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),  # Apply standard normalization
        ])

    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.apply_augmentation:
            # Apply augmentation by concatenating and reshaping
            image_label = torch.cat([image, label], 0)
            image_label = self.shape_transform(image_label)
            image, label = torch.split(image_label, [3, 1], dim=0)  # Split back into image and label
            image = self.color_transform(image)

        image = self.normalize(image)
        label = label.to(torch.int64)
        label = torch.squeeze(label, dim=0)
        return image, label


class ValidationDataset(torch.utils.data.Dataset):
    """
    Dataset class for validation, applying only normalization.
    """

    def __init__(self, images, labels):
        # Initialize image and label data
        self.images = np.transpose(images, axes=(0, 3, 1, 2))
        self.images = torch.from_numpy(self.images)
        self.labels = np.expand_dims(labels, axis=1)
        self.labels = torch.from_numpy(self.labels)
        self.datanum = self.images.size(0)

        # Define normalization for validation
        self.normalize = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32) / 255),  # Normalize to [0, 1]
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),  # Apply standard normalization
        ])

    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image = self.normalize(image)
        label = label.to(torch.int64)
        label = torch.squeeze(label, dim=0)

        return image, label


class TestDataset(torch.utils.data.Dataset):
    """
    Dataset class for testing, applying only normalization.
    """

    def __init__(self, images):
        # Initialize image data
        self.images = np.transpose(images, axes=(0, 3, 1, 2))
        self.images = torch.from_numpy(self.images)
        self.datanum = self.images.size(0)

        self.normalize = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32) / 255),  # Normalize to [0, 1]
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),  # Apply standard normalization
        ])

    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        image = self.images[index]
        image = self.normalize(image)

        return image
