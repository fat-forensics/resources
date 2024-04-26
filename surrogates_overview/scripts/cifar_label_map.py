"""
CIFAR 10 & 100 Labels Map
=========================

This module provides a map from a class id to label.
Two maps are available:

* ``CIFAR10_LABEL_MAP`` -- maps class ids to labels for CIFAR10; and
* ``CIFAR100_LABEL_MAP`` -- maps class ids to labels for CIFAR100.

The data set files needed to regenerate the label maps are available at
<https://www.cs.toronto.edu/~kriz/cifar.html>.

See <https://github.com/fat-forensics/resources/tree/master/surrogates_overview>
for more details.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD


import pickle


def _load_cifar10_labels(data_folder):
    """Generates the label map for CIFAR10."""
    with open(f'{data_folder}/cifar-10-batches-py/batches.meta', 'rb') as fo:
        cf10meta = pickle.load(fo, encoding='bytes')

    cf10labels = {i: j.decode()
                  for i, j in enumerate(cf10meta.get(b'label_names'))}

    return cf10labels


def _load_cifar100_labels(data_folder, fine_labels=True):
    """Generates the label map for CIFAR100."""
    with open(f'{data_folder}/cifar-100-python/meta', 'rb') as fo:
        cf100meta = pickle.load(fo, encoding='bytes')

    if fine_labels:
        cf100_labels_type = b'fine_label_names'
    else:
        cf100_labels_type = b'coarse_label_names'

    cf100labels = {i: j.decode()
                  for i, j in enumerate(cf100meta.get(cf100_labels_type))}

    return cf100labels


CIFAR10_LABEL_MAP = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


CIFAR100_LABEL_MAP = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    3: 'bear',
    4: 'beaver',
    5: 'bed',
    6: 'bee',
    7: 'beetle',
    8: 'bicycle',
    9: 'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'crab',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}
