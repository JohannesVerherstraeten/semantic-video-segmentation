"""
Heavily based on from https://github.com/davidtvs/PyTorch-ENet
"""

import numpy as np
import torch
import os


def enet_weighing(dataset, ignore_labels=(), c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    :type dataset: data.dataset.basedataset.BaseDataset
    :type ignore_labels: tuple | list
    :type c: float
    """
    path = dataset.get_base_directory() + "enet_weights.wht"
    if os.path.exists(path):
        class_weights = torch.load(path)
        return class_weights

    ignore_indices = [list(dataset.get_color_encoding()).index(label) for label in ignore_labels]
    class_count = 0
    total = 0

    def class_count_iteration(label, class_count, total):
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=dataset.get_nb_classes())
        total += flat_label.size
        return class_count, total

    for label in dataset.get_labels():
        if label is not None:
            class_count, total = class_count_iteration(label, class_count, total)

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    # Set the weight of the classes to ignore to 0
    class_weights[ignore_indices] = 0
    class_weights = torch.from_numpy(class_weights).float()

    torch.save(class_weights, path)

    return class_weights


def median_freq_balancing(dataset, ignore_labels=()):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

        w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    :type dataset: data.dataset.basedataset.BaseDataset
    :type ignore_labels: tuple | list
    """
    ignore_indices = [list(dataset.get_color_encoding()).index(label) for label in ignore_labels]
    class_count = 0
    total = 0

    def class_count_iteration(label, class_count, total):
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=dataset.get_nb_classes())

        # Create mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount
        return class_count, total

    for label in dataset.get_labels():
        if label is not None:
            class_count, total = class_count_iteration(label, class_count, total)

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    class_weights = med / freq

    # Set the weight of the classes to ignore to 0
    class_weights[ignore_indices] = 0
    class_weights = torch.from_numpy(class_weights).float()

    return class_weights
