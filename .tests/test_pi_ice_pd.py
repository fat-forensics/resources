"""
Test PFI, ICE and PD
====================
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import numpy as np
import pi_ice_pd as fns

import sklearn.datasets
import sklearn.linear_model

# Load the iris data set
IRIS = sklearn.datasets.load_iris()
IRIS_X = IRIS.data  # [:, :2]  #  take the first two features only
IRIS_Y = IRIS.target
IRIS_FEATURE_NAMES = IRIS.feature_names

# Fit the classifier
LOGREG = sklearn.linear_model.LogisticRegression(C=1e5, random_state=42)
LOGREG.fit(IRIS_X, IRIS_Y)

PFI = {
    'r2': [
        ('sepal length (cm)', 0.0268, 0.0140),
        ('sepal width (cm)', 0.0520, 0.0190),
        ('petal length (cm)', 1.3553, 0.1263),
        ('petal width (cm)', 0.3611, 0.0426)
    ],
    'neg_mean_squared_error': [
        ('sepal length (cm)', 0.0179, 0.0093),
        ('sepal width (cm)', 0.0347, 0.0126),
        ('petal length (cm)', 0.9035, 0.0842),
        ('petal width (cm)', 0.2407, 0.0284)
    ],
    'neg_mean_absolute_error': [
        ('sepal length (cm)', 0.0179, 0.0093),
        ('sepal width (cm)', 0.0347, 0.0126),
        ('petal length (cm)', 0.6738, 0.0511),
        ('petal width (cm)', 0.2407, 0.0284)
    ],
    'max_error': [
        ('sepal length (cm)', 0.0, 0.0),
        ('sepal width (cm)', 0.0, 0.0),
        ('petal length (cm)', 1.0, 0.0),
        ('petal width (cm)', 0.0, 0.0)
    ]
}

ICE_PD = {
    0: {
        'ice': np.array([[[1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0]],

                         [[0.9999, 0.0, 0.0],
                          [0.9999, 0.0, 0.0],
                          [0.9999, 0.0, 0.0],
                          [1.0000, 0.0, 0.0],
                          [1.0000, 0.0, 0.0]],
                  
                         [[0.0, 0.9984, 0.0016],
                          [0.0, 0.9996, 0.0004],
                          [0.0, 0.9999, 0.0001],
                          [0.0, 0.9999, 0.0000],
                          [0.0, 0.9999, 0.0000]],
                  
                         [[0.0, 0.9996, 0.0004],
                          [0.0, 0.9999, 0.0001],
                          [0.0, 0.9999, 0.0000],
                          [0.0, 0.9999, 0.0000],
                          [0.0, 0.9999, 0.0000]],
                  
                         [[0.0, 0.0, 1.0000],
                          [0.0, 0.0, 1.0000],
                          [0.0, 0.0, 1.0000],
                          [0.0, 0.0, 0.9999],
                          [0.0, 0.0, 0.9999]],
                  
                         [[0.0, 0.0000, 0.9999],
                          [0.0, 0.0000, 0.9999],
                          [0.0, 0.0000, 0.9999],
                          [0.0, 0.0001, 0.9999],
                          [0.0, 0.0004, 0.9996]]]),
        'pd': np.array([[0.3333, 0.3330, 0.3337],
                        [0.3333, 0.3332, 0.3334],
                        [0.3333, 0.3333, 0.3334],
                        [0.3333, 0.3333, 0.3333],
                        [0.3333, 0.3334, 0.3333]]),
        'linspace': np.array([5.0, 5.55, 6.1, 6.65, 7.2])
    }
}


def test_build_permutation_importance():
    """Tests permutation feature importance."""
    metrics = [
        'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'max_error'
    ]
    pfi = fns.build_permutation_importance(
        IRIS_X, IRIS_Y, IRIS_FEATURE_NAMES, LOGREG, metrics)

    assert len(pfi.keys()) == len(metrics)
    assert len(pfi.keys()) == len(PFI.keys())
    for m in metrics:
        assert m in pfi
        assert m in PFI
        assert len(pfi[m]) == len(PFI[m])
        for i, j in zip(sorted(pfi[m]), sorted(PFI[m])):
            assert i[0] == j[0]
            assert np.allclose(i[1], j[1], atol=.001, equal_nan=True)
            assert np.allclose(i[2], j[2], atol=.001, equal_nan=True)


def test_build_ice_pd():
    """Tests individual conditional expectation and partial dependence."""
    instance_indices = [0, 25, 50, 75, 100, 125]
    feature_indices = [0]
    ice_pd = fns.build_ice_pd(
        IRIS_X[instance_indices, :],
        LOGREG,
        feature_indices,
        samples_no=5)

    assert len(ice_pd.keys()) == len(feature_indices)
    assert len(ice_pd.keys()) == len(ICE_PD.keys())
    for m in feature_indices:
        assert m in ice_pd
        assert m in ICE_PD
        assert len(ice_pd[m]) == len(ICE_PD[m])

        assert 'linspace' in ice_pd[m]
        assert 'ice' in ice_pd[m]
        assert 'pd' in ice_pd[m]

        assert 'linspace' in ICE_PD[m]
        assert 'ice' in ICE_PD[m]
        assert 'pd' in ICE_PD[m]

        assert np.allclose(
            ice_pd[m]['linspace'],
            ICE_PD[m]['linspace'],
            atol=.001,
            equal_nan=True)
        assert np.allclose(
            ice_pd[m]['ice'],
            ICE_PD[m]['ice'],
            atol=.001,
            equal_nan=True)
        assert np.allclose(
            ice_pd[m]['pd'],
            ICE_PD[m]['pd'],
            atol=.001,
            equal_nan=True)
