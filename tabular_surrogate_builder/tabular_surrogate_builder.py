"""
Tabular Surrogate Explainer Builder
===================================

This module implements a collection of helper functions for an interactive
builder of surrogate explainers of tabular data based on iPyWidgets.
See <https://github.com/fat-forensics/resources/tree/master/tabular_surrogate_builder>
for more details.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

from sklearn.linear_model import RidgeClassifier

import numpy as np

import ipywidgets as widgets
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def build_tabular_blimey(
        instance,
        class_to_explain,
        sampled_data,
        prediction_fn,
        discretisation,
        fit_intercept=True,
        random_seed=42
    ):
    """
    Composes a tabular bLIMEy surrogate explainer based on ridge classification.
    """
    preds = (prediction_fn(sampled_data) == class_to_explain).astype(np.int8)

    # digitize data
    data_dig = np.vstack([
        np.digitize(sampled_data[:, 0], discretisation[0]),
        np.digitize(sampled_data[:, 1], discretisation[1])
    ]).T
    # digitize point
    point_dig = np.array([
        np.digitize(instance[0], discretisation[0]),
        np.digitize(instance[1], discretisation[1])
    ])
    #
    binary_data = (data_dig == point_dig).astype(np.int8)
    # np.unique(binary_data, axis=0)

    # train ridge
    clf = RidgeClassifier(fit_intercept=fit_intercept,
                          random_state=random_seed)
    clf.fit(binary_data, preds)
    # return coefficients
    return clf.coef_[0]


def plot_tabular_explanation(
        instance,
        class_to_explain,
        sampled_data,
        prediction_fn,
        discretisation,
        feature_ranges,
        feature_names,
        explanation
    ):
    """Plots a tabular bLIMEy explanation."""
    explanation = explanation.copy() / np.abs(explanation).sum()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_alpha(0)
    fig.suptitle('Explained class: {}'.format(class_to_explain), fontsize=18)

    # plot /petal length (cm)/ vs /petal width/
    # x_name, y_name = 'petal length (cm)', 'petal width (cm)'
    # x_ind, y_ind = iris.feature_names.index(x_name), iris.feature_names.index(y_name)
    x_min, x_max = feature_ranges[0][0] - .5, feature_ranges[0][1] + .5
    y_min, y_max = feature_ranges[1][0] - .5, feature_ranges[1][1] + .5
    #
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    #plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    Z = prediction_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax_l.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    #
    ax_l.scatter(sampled_data[:, 0],
                 sampled_data[:, 1],
                 c=prediction_fn(sampled_data),
                 cmap=plt.cm.Set1, edgecolor='k')
    ax_l.set_xlabel(feature_names[0], fontsize=18)
    ax_l.set_ylabel(feature_names[1], fontsize=18)
    #
    ax_l.set_xlim(x_min, x_max)
    ax_l.set_ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    ax_l.scatter(instance[0], instance[1],
                 c='yellow', marker='*', s=500, edgecolor='k')
    ax_l.vlines(discretisation[0], -1, 10, linewidth=3)
    ax_l.hlines(discretisation[1], -1, 10, linewidth=3)
    #
    ax_l.tick_params(axis='x', labelsize=18)
    ax_l.tick_params(axis='y', labelsize=18)

    x_dig_ = np.digitize(instance[0], discretisation[0])
    x_dig_list_ = ['-inf'] + [str(i) for i in discretisation[0]] + ['+inf']
    y_dig_ = np.digitize(instance[1], discretisation[1])
    y_dig_list_ = ['-inf'] + [str(i) for i in discretisation[1]] + ['+inf']
    x = ['{}\n{} < ... <= {}'.format(feature_names[0],
                                     x_dig_list_[x_dig_],
                                     x_dig_list_[x_dig_ + 1]),
         '{}\n{} < ... <= {}'.format(feature_names[1],
                                     y_dig_list_[y_dig_],
                                     y_dig_list_[y_dig_ + 1])]
    #
    y = [abs(i) for i in explanation]
    c = ['green' if i >= 0 else 'red' for i in explanation]

    ax_r.set_xlim([0, 1.20])
    ax_r.set_ylim([-.5, len(x) - .5])
    ax_r.grid(False, axis='y')
    ax_r.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax_r.barh(x, y, height=.5, color=c)
    ax_r.set_yticklabels([])
    for i, v in enumerate(y):
        ax_r.text(v + .02, i + .15, '{:.4f}'.format(v),
                  fontweight='bold', fontsize=18)
        ax_r.text(v + .02, i - .2, x[i], fontweight='bold', fontsize=18)

    # highlight explained spot
    x_dig_list_val_ = [0] + [i for i in discretisation[0]] + [8]
    ax_l.axvspan(x_dig_list_val_[x_dig_], x_dig_list_val_[x_dig_ + 1],
                 facecolor='None', hatch='/', alpha=1.0)
    y_dig_list_val_ = [-.5] + [i for i in discretisation[1]] + [3.5]
    ax_l.axhspan(y_dig_list_val_[y_dig_], y_dig_list_val_[y_dig_ + 1],
                 facecolor='None', hatch='\\', alpha=1.0)
    
    ax_r.tick_params(axis='x', labelsize=18)
    # ax_r.tick_params(axis='y', labelsize=18)

    plt.show()


def _generate_data(data_samples_no, x_range, y_range, random_seed):
    """
    Generates a random data sample with a fixed number of instances per split.
    """
    x_range_n, y_range_n = len(x_range), len(y_range)
    assert x_range_n > 2 and y_range_n > 2

    y_range_ = y_range[::-1]
    data_, i = [], 0
    for y_ in range(y_range_n - 1):
        for x_ in range(x_range_n - 1):
            np.random.seed(random_seed)
            d_ = np.random.uniform(
                low=(x_range[x_] + 0.1, y_range_[y_ + 1] + 0.1),
                high=(x_range[x_ + 1] - 0.1, y_range_[y_] - 0.1),
                size=(data_samples_no[i], 2)
            )
            data_.append(d_)
            i += 1
    data_ = np.vstack(data_)
    return data_


def generate_tabular_widget(
        black_boxes,
        class_map,
        feature_specification,
        random_seed=42
    ):
    """Builds iPyWidget interactive tabular surrogate explainer."""

    def explain_action(obj):
        prediction_fn_ = black_boxes[lime_bb_toggle.value]

        instance_ =  np.array(
            [lime_instance_toggle.children[0].value,
             lime_instance_toggle.children[1].value],
            dtype=np.float64
        )

        feature_names_ = {
            0: feature_specification[0]['name'],
            1: feature_specification[1]['name']
        }

        feature_ranges_ = {
            0: feature_specification[0]['range'],
            1: feature_specification[1]['range']
        }

        x_axis_range_ = x_axis_slider.value
        assert x_axis_range_[0] != x_axis_range_[1], (
            'The petal length split values must not be identical.')
        y_axis_range_ = y_axis_slider.value
        assert y_axis_range_[0] != y_axis_range_[1], (
            'The petal width split values must not be identical.')
        discretisation_ = {0: x_axis_range_, 1: y_axis_range_}

        data_samples_no = [i.value for i in sample_widget.children]
        x_range_ = [feature_specification[0]['range'][0],
                    x_axis_range_[0],
                    x_axis_range_[1],
                    feature_specification[0]['range'][1]]
        y_range_ = [feature_specification[1]['range'][0],
                    y_axis_range_[0],
                    y_axis_range_[1],
                    feature_specification[1]['range'][1]]
        data_ = _generate_data(data_samples_no, x_range_, y_range_, random_seed)

        explained_class_ = lime_class_toggle.value
        explained_class_id_ = class_map[explained_class_]

        explanation_ = build_tabular_blimey(
            instance_,
            explained_class_id_,
            data_,
            prediction_fn_,
            discretisation_,
            fit_intercept=model_intercept_widget.value,
            random_seed=random_seed
        )

        with lime_explain_out:
            lime_explain_out.clear_output(wait=True)
            plot_tabular_explanation(
                instance_,
                explained_class_,
                data_,
                prediction_fn_,
                discretisation_,
                feature_ranges_,
                feature_names_,
                explanation_
            )
            plt.show()

    # Explained instance -- select 1 of three points
    lime_instance_toggle_items = [
        widgets.BoundedFloatText(
            value=feature_specification[0]['instance']['value'],
            min=feature_specification[0]['range'][0],
            max=feature_specification[0]['range'][1],
            step=feature_specification[0]['instance']['step'],
            description='[X] {}:'.format(feature_specification[0]['name']),
            disabled=False
        ),
        widgets.BoundedFloatText(
            value=feature_specification[1]['instance']['value'],
            min=feature_specification[1]['range'][0],
            max=feature_specification[1]['range'][1],
            step=feature_specification[1]['instance']['step'],
            description='[Y] {}:'.format(feature_specification[1]['name']),
            disabled=False
        )
    ]
    lime_instance_toggle = widgets.GridBox(
        lime_instance_toggle_items,
        layout=widgets.Layout(grid_template_columns='repeat(2, 315px)')
    )

    # Select a class to explain
    lime_class_toggle = widgets.ToggleButtons(
        options=list(class_map.keys()),
        description='Class:',
        disabled=False,
        button_style=''  # 'success', 'info', 'warning', 'danger' or ''
        # tooltips=['Description of slow', 'Description of regular',
        #           'Description of fast'],
        # icons=['check'] * 3
    )

    items_sample_widget = [
        widgets.BoundedIntText(
            value=100,
            min=0,
            max=1000,
            step=1,
            description='Samples #:',
            disabled=False
        )
        for _ in range(9)
    ]
    sample_widget = widgets.GridBox(
        items_sample_widget,
        layout=widgets.Layout(grid_template_columns='repeat(3, 300px)'))

    # select two thresholds for the segmentation
    step_ = str(feature_specification[0]['discretisation']['step'])
    prec_ = len(step_.split('.')[1]) if '.' in step_ else 0
    x_axis_slider = widgets.FloatRangeSlider(
        value=feature_specification[0]['discretisation']['init'],
        min=feature_specification[0]['discretisation']['range'][0],
        max=feature_specification[0]['discretisation']['range'][1],
        step=feature_specification[0]['discretisation']['step'],
        description='[X] {}:'.format(feature_specification[0]['name']),
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.{}f'.format(prec_)
    )
    step_ = str(feature_specification[1]['discretisation']['step'])
    prec_ = len(step_.split('.')[1]) if '.' in step_ else 0
    y_axis_slider = widgets.FloatRangeSlider(
        value=feature_specification[1]['discretisation']['init'],
        min=feature_specification[1]['discretisation']['range'][0],
        max=feature_specification[1]['discretisation']['range'][1],
        step=feature_specification[1]['discretisation']['step'],
        description='[Y] {}:'.format(feature_specification[1]['name']),
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.{}f'.format(prec_)
    )

    model_intercept_widget = widgets.Checkbox(
        value=True,
        description='Model intercept?',
        disabled=False,
        indent=False
    )
    lime_bb_toggle = widgets.ToggleButtons(
        options=list(black_boxes.keys()),
        description='Black box:',
        disabled=False,
        button_style=''
    )

    lime_explain_button = widgets.Button(
        description='Explain!',
        disabled=False,
        button_style='info',
        tooltip='Explain',
        icon='check'
    )

    lime_explain_out = widgets.Output()

    lime_explain_button.on_click(explain_action)
    # pre-click the button
    lime_explain_button._click_handlers(lime_explain_button)

    interactive_explainer = widgets.VBox([
        lime_instance_toggle,
        lime_class_toggle,
        sample_widget,
        x_axis_slider,
        y_axis_slider,
        model_intercept_widget,
        lime_bb_toggle,
        lime_explain_button,
        lime_explain_out
    ])

    return interactive_explainer
