"""
Image Classifier
================

This module implements an image classifier based on PyTorch.
Inception v3 and AlexNet are availabel.
See <https://github.com/fat-forensics/resources/tree/master/surrogates_overview>
for more details.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from scripts.imagenet_label_map import IMAGENET_LABEL_MAP

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


def _get_preprocess_transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([transforms.ToTensor(), normalize])

    return transf


class ImageClassifier(object):
    """Image classifier based on PyTorch."""

    def __init__(self, model='inception_v3'):
        """Initialises the image classifier."""
        assert model in ('inception_v3', 'alexnet')

        # Get class labels
        self.class_idx = IMAGENET_LABEL_MAP

        # Get the model
        if model == 'inception_v3':
            clf = models.inception_v3(pretrained=True, transform_input=False)
        else:
            clf = models.alexnet(pretrained=True)
        self.clf = clf
        self.clf.eval()

        # Get transformation
        self.preprocess_transform = _get_preprocess_transform()

    def fit(self, _X, _y):
        """Fits the image classifier -- a dummy method."""
        return

    def predict_proba(self, X):
        """Predicts probabilities of the collection of images."""
        X_ = [self.preprocess_transform(x) for x in X]
        tensor = torch.stack(X_, dim=0)
        prediction = F.softmax(self.clf(tensor), dim=1)
        return prediction.detach().cpu().numpy()

    def predict(self, X, labels=False):
        """Predicts class indices of the collection of images."""
        prediction = self.predict_proba(X)
        prediction_idxs = np.argmax(prediction, axis=1)
        if labels:
            classes = [self.class_idx[i] for i in prediction_idxs]
        else:
            classes = prediction_idxs
        return classes

    def proba2prediction(self, Y):
        """Converts predicted probabilities to class indices."""
        classes = np.argmax(Y, axis=1)
        return classes

    def prediction2label(self, Y):
        """Converts class indices to label names."""
        labels = [self.class_idx[y] for y in Y]
        return labels

    def proba2label(self, Y, labels_no=5):
        """Converts class probabilities to label names."""
        ordered_classes = np.flip(np.argsort(Y, axis=1))
        ordered_classes_top = ordered_classes[:, :labels_no]
        labels_top = np.vectorize(self.class_idx.get)(ordered_classes_top)
        return labels_top

    def proba2tuple(self, Y, labels_no=5):
        """Converts class probabilities to (label name, probability) tuples."""
        ordered_classes = np.flip(np.argsort(Y, axis=1))

        ordered_classes_top = ordered_classes[:, :labels_no]
        labels_top = np.vectorize(self.class_idx.get)(ordered_classes_top)

        tuples = []
        for idx in range(Y.shape[0]):
            tuples_ = []
            for cls, lab in zip(ordered_classes_top[idx], labels_top[idx]):
                tuples_.append((lab, Y[idx, cls], cls))
            tuples.append(tuples_)
        return tuples
