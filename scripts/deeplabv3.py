import os
import pathlib
import sys
import time

import numpy as np
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

from dataset import TrainDataset, ValidationDataset, TestDataset
from loss import TissueLoss, NucleusLoss


class Config:
    """Configuration class for model training settings"""

    def __init__(
            self,
            patch_shape,
            num_classes,
            learning_rate,
            train_batch_size,
            validation_batch_size,
            segmentation_type,
            iteration_count
    ):
        # Patch shape for segmentation
        self.patch_shape = patch_shape
        # Number of segmentation classes
        self.num_classes = num_classes
        # Learning rate for the optimizer
        self.learning_rate = learning_rate
        # Batch size for training
        self.train_batch_size = train_batch_size
        # Batch size for validation
        self.validation_batch_size = validation_batch_size
        # Type of segmentation (tissue or nucleus)
        self.segmentation_type = segmentation_type
        # Total iteration count for training
        self.iteration_count = iteration_count
        # Large batch size for specific training operations
        self.large_batch_size = train_batch_size * 100


def mkdir(dirname):
    """Create a directory if it doesn't exist"""
    os.makedirs(pathlib.Path(dirname).resolve(), exist_ok=True)


def set_random_seed(seed):
    """Set the random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_DeepLabv3(outputchannels=1):
    """Create a DeepLabV3 model with custom classifier for segmentation"""
    # Load the default DeepLabV3 model
    model = models.segmentation.deeplabv3_resnet101(
        weights=models.segmentation.deeplabv3.DeepLabV3_ResNet101_Weights.DEFAULT,
        progress=True)
    # Customize the classifier with a different output channel count
    model.classifier = DeepLabHead(2048, outputchannels)
    model.aux_classifier = models.segmentation.fcn.FCNHead(
        1024, outputchannels)
    return model


def train(
        config,
        custom_train_data_loader,
        label_weights,
        validation_images,
        validation_labels,
        checkpoint_prefix,
        random_seed=3141592653
):
    # Set the random seed for consistent results
    set_random_seed(random_seed)

    # Create a DeepLabV3 model with the specified number of classes
    model = create_DeepLabv3(outputchannels=config.num_classes)

    # Prepare the validation data loader
    validation_data_loader = torch.utils.data.DataLoader(
        ValidationDataset(validation_images, validation_labels),
        batch_size=config.validation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1
    )
    del validation_images, validation_labels

    # Choose a computing device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # If multiple GPUs are available, use data parallelism
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Set up the optimizer and loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Use different loss functions depending on segmentation type
    if config.segmentation_type == 'tissue':
        criterion = TissueLoss(config.num_classes, label_weights).to(device)
    elif config.segmentation_type == 'nucleus':
        criterion = NucleusLoss().to(device)

    # Load the training data
    train_images, train_labels = custom_train_data_loader.load_data(
        config.large_batch_size
    )
    train_data_loader = torch.utils.data.DataLoader(
        TrainDataset(train_images, train_labels, config.patch_shape),
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1
    )
    del train_images, train_labels

    # Initialize iterator and set training mode
    iterator = iter(train_data_loader)
    best_validation_IoU = 0  # Best IoU score
    start_time = time.time()  # Start timer for training
    model.train()

    # Training loop for a specified number of iterations
    for step in range(1, config.iteration_count + 1):
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(train_data_loader)
                images, labels = next(iterator)
            images = images.to(device)
            outputs = model(images)
            labels = labels.to(device)
            loss = criterion(outputs['out'], labels)
            loss += 0.4 * criterion(outputs['aux'], labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        # Periodic output every 100 iterations
        if step % 100 == 0:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            print('Elapsed time: {:f} [s]'.format(elapsed_time))
            log_items = []
            log_items.append('Step {:5d}'.format(step))
            log_items.append('Training loss = {:.7f}'.format(loss.item()))

            # Calculate validation loss and scores
            validation_loss, recalls, precisions, IoUs = get_validation_scores(
                model, validation_data_loader, config.num_classes, criterion,
                device
            )
            log_items.append(
                'Validation loss = {:.7f}'.format(validation_loss)
            )
            print(', '.join(log_items))

            # Calculate mean IoU and save the best model
            mean_IoU = np.mean(IoUs[1:])
            for i in range(1, config.num_classes):
                print('Class: {:d}'.format(i + 1))
                print('     Recall: {:.7f}'.format(recalls[i]))
                print('  Precision: {:.7f}'.format(precisions[i]))
                print('        IoU: {:.7f}'.format(IoUs[i]))
            print('   mean IoU: {:.7f}'.format(mean_IoU))

            # Save the model with the best IoU
            if mean_IoU >= best_validation_IoU:
                best_validation_IoU = mean_IoU
                checkpoint_file = checkpoint_prefix + '_IoU_best.pth'  # File name for best model
                mkdir(os.path.dirname(checkpoint_file))
                torch.save(model.state_dict(), checkpoint_file)
                print('Checkpoint has been saved to ' + checkpoint_file)

            # Reset and reload data
            sys.stdout.flush()
            start_time = time.time()  # Restart timer
            train_images, train_labels = custom_train_data_loader.load_data(
                config.large_batch_size
            )
            train_data_loader = torch.utils.data.DataLoader(
                TrainDataset(train_images, train_labels, config.patch_shape),
                batch_size=config.train_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=1
            )
            del train_images, train_labels
            iterator = iter(train_data_loader)
            model.train()

        # Save checkpoints every 5000 iterations
        if step % 5000 == 0 and step >= 10000:
            checkpoint_file = '{:s}_{:d}.pth'.format(checkpoint_prefix, step)  # Checkpoint filename
            torch.save(model.state_dict(), checkpoint_file)
            print(
                'Checkpoint has been saved to ' + checkpoint_file, flush=True
            )


def get_validation_scores(model, data_loader, num_classes, criterion, device):
    """
    Calculate validation scores, including loss, recalls, precisions, and IoUs.
    """
    true_counts = np.zeros(num_classes, np.int32)  # Ground truth counts
    positive_counts = np.zeros(num_classes, np.int32)  # Predicted positive counts
    true_positive_counts = np.zeros(num_classes, np.int32)  # True positive counts
    loss = 0  # Initialize loss
    iteration_count = 0  # Initialize iteration count
    model.eval()

    # Iterate over the validation data
    with torch.no_grad():
        for images, labels in iter(data_loader):
            images = images.to(device)
            outputs = model(images)
            labels = labels.to(device)
            logits = outputs['out']
            loss += criterion(logits, labels).item()
            y_pred = torch.nn.functional.one_hot(
                torch.softmax(logits, 1).argmax(1), num_classes=num_classes
            )
            y_pred = torch.reshape(y_pred, [-1, num_classes])
            y_true = torch.nn.functional.one_hot(
                labels, num_classes=num_classes
            )
            y_true = torch.reshape(y_true, [-1, num_classes])
            true_counts += torch.sum(y_true, dim=0).cpu().numpy()
            positive_counts += torch.sum(y_pred, dim=0).cpu().numpy()
            true_positive_counts += torch.sum(
                y_pred * y_true, dim=0).cpu().numpy()
            iteration_count += 1

        del images, outputs, labels, logits
        torch.cuda.empty_cache()

    # Calculate recall, precision, and IoU scores
    recalls = true_positive_counts / true_counts
    precisions = true_positive_counts / positive_counts
    unions = true_counts + positive_counts - true_positive_counts
    IoUs = true_positive_counts / unions
    loss /= iteration_count  # Normalize loss by iteration count
    return loss, recalls, precisions, IoUs


class Decoder:
    """
    Decoder for performing inference with a trained DeepLabV3 model.
    """

    def __init__(self, checkpoint_file, num_classes):
        # Create and configure the model
        self.model = create_DeepLabv3(outputchannels=num_classes)
        state_dict = torch.load(checkpoint_file)
        self.model.load_state_dict(state_dict)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def decode(self, images, batch_size, buffer_size=10000):
        """
        Decode the given images using the trained DeepLabV3 model.
        """
        self.model.eval()
        predicted_label_list = []
        with torch.no_grad():
            num_images = len(images)
            index = 0
            while index < num_images:
                next_index = min(index + buffer_size, num_images)
                data_loader = torch.utils.data.DataLoader(
                    TestDataset(np.array(images[index:next_index])),
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=1
                )

                # Process each batch in the data loader
                for images in iter(data_loader):
                    images = images.to(self.device)
                    outputs = self.model(images)
                    prob_maps = torch.softmax(
                        outputs['out'], 1).permute(0, 2, 3, 1).cpu().numpy()
                    predicted_label_list.extend(prob_maps)

                index = next_index

        return predicted_label_list
