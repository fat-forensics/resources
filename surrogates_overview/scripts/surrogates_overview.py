"""
Surrogate Explainability Overview Helper Functions
==================================================

This library implements a collection of helper functions for a range of
explainability methods.
Among others, it implements interactive iPyWidgets for no-code experimenting
with AI and ML explainers.
See <https://github.com/fat-forensics/resources/tree/master/surrogates_overview>
for more details.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import fatf.utils.data.instance_augmentation as fatf_instance_augmentation
import fatf.utils.data.augmentation as fatf_data_augmentation
import fatf.utils.data.occlusion as fatf_occlusion
import fatf.utils.data.segmentation as fatf_segmentation
import fatf.utils.models.processing as fatf_processing
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

import fatf
import sklearn.linear_model


def plot_prediction(tuple_list, ax=None):
    """Plots a predictions."""
    x = [i[0] for i in tuple_list[::-1]]
    y = [i[1] for i in tuple_list[::-1]]

    if ax is None:
        ax = plt
        ax.figure(figsize=(4, 4))
        ax.xlim([0, 1.20])
        ax.ylim([-.5, len(x) - .5])
    else:
        ax.set_xlim([0, 1.30])
        ax.set_ylim([-.5, len(x) - .5])
        ax.grid(False, axis='y')
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.barh(x, y, height=.5)
    for i, v in enumerate(y):
        ax.text(v + .02,
                i + .0,
                '{:.4f}'.format(v),
                fontweight='bold',
                fontsize=18)


def plot_image_prediction(prediction, image):
    """Displays a bar-plot visualisation of a prediction next to an image."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_alpha(0)
    plot_prediction(prediction, ax_l)
    ax_l.tick_params(axis='x', labelsize=18)
    ax_l.tick_params(axis='y', labelsize=18)
    ax_r.imshow(image)
    ax_r.grid(False)
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    return fig


def build_image_blimey(image,
                       prediction_fn,
                       explain_label_class,
                       explanation_size=5,
                       segments_number=13,
                       occlusion_colour='mean',
                       samples_number=50,
                       batch_size=50,
                       random_seed=42):
    """Builds a bLIMEy surrogate image explainer."""
    segmenter = fatf_segmentation.Slic(
        image, n_segments=segments_number)
    occluder = fatf_occlusion.Occlusion(
        image, segmenter.segments, colour=occlusion_colour)

    fatf.setup_random_seed(random_seed)
    sampled_data = fatf_instance_augmentation.random_binary_sampler(
        segmenter.segments_number, samples_number)

    iter_ = fatf_processing.batch_data(
        sampled_data,
        batch_size=batch_size,
        transformation_fn=occluder.occlude_segments_vectorised)
    sampled_data_probabilities = []
    for batch in iter_:
        batch_predictions = prediction_fn(batch)
        sampled_data_probabilities.append(batch_predictions)
    sampled_data_probabilities = np.vstack(sampled_data_probabilities)

    surrogates = {}
    # Explain each class with a ridge regression
    for class_label, class_id in explain_label_class:
        class_probs = sampled_data_probabilities[:, class_id]

        surrogate = sklearn.linear_model.Ridge(
            alpha=1, fit_intercept=True, random_state=random_seed)
        surrogate.fit(sampled_data, class_probs)

        feature_ordering = np.flip(np.argsort(np.abs(surrogate.coef_)))
        top_features = feature_ordering[:explanation_size]

        explanation = list(zip(top_features, surrogate.coef_[top_features]))
        
        surrogates[class_id] = dict(
            name=class_label,
            model=surrogate,
            explanation=explanation)

    explainers = dict(
        segmenter=segmenter,
        occluder=occluder,
        sampled_data=sampled_data,
        sampled_data_probabilities=sampled_data_probabilities,
        surrogates=surrogates
    )
    return explainers


def plot_image_explanation(blimey, explained_class, show_random=False):
    """
    Plots a bar-plot explanation, image-colouring explanation and image
    segmentation/random occlusion sample triplet.
    """
    class_id = explained_class[1]

    title = blimey['surrogates'][class_id]['name']
    assert title == explained_class[0]
    explanation = blimey['surrogates'][class_id]['explanation']

    occluder = blimey['occluder']
    segmenter = blimey['segmenter']

    fig, (ax_l, ax_r, ax_rr) = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_alpha(0)
    plt.suptitle(title, fontsize=18)

    # Show bar-plot explanation
    x = ['#' + str(i[0] + 1) for i in explanation[::-1]]
    y = [abs(i[1]) for i in explanation[::-1]]
    c = ['green' if i[1] >= 0 else 'red' for i in explanation[::-1]]

    ax_l.set_xlim([0, 1.00])
    ax_l.set_ylim([-.5, len(x) - .5])
    ax_l.grid(False, axis='y')
    ax_l.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax_l.tick_params(axis='x', labelsize=18)
    ax_l.tick_params(axis='y', labelsize=18)

    ax_l.barh(x, y, height=.5, color=c)
    for i, v in enumerate(y):
        ax_l.text(v + .02,
                  i + .0,
                  '{:.4f}'.format(v),
                  fontweight='bold',
                  fontsize=18)

    # number and highlight top segments
    segment_id, colour = [], []
    for feature_id, importance in explanation:
        segment_id.append(int(feature_id + 1))
        colour.append('r' if importance < 0 else 'g')
    highlighted = segmenter._stain_segments(
        segments_subset=segment_id, colour=colour)
    numbered = segmenter.number_segments(
        image=highlighted, segments_subset=segment_id, colour=(255, 255, 0))
    ax_r.imshow(numbered)
    ax_r.grid(False)
    ax_r.set_xticks([])
    ax_r.set_yticks([])

    # show segmentation or a random instance
    if show_random:
        right_most = occluder.occlude_segments(show_random)
        right_most = segmenter.mark_boundaries(
            image=right_most, colour=(255, 255, 0))
    else:
        right_most = segmenter.mark_boundaries(colour=(255, 255, 0))
        right_most = segmenter.number_segments(
            image=right_most, colour=(255, 0, 0))
    ax_rr.imshow(right_most)
    ax_rr.grid(False)
    ax_rr.set_xticks([])
    ax_rr.set_yticks([])

    plt.show()


def generate_image_widget(
        blimey_collection,
        granularity_selection,
        colour_selection,
        class_selection):
    """Creates an interactive bLIMEy image explanation widget."""

    def explain_action(obj):
        gran = granularity_selection_slider.value
        col = colour_selection_toggle.value

        explainer = blimey_collection[gran][col]
        explain_class = (class_selection_toggle.value,
                         class_selection[class_selection_toggle.value])
        random_idx = np.random.randint(
            0, explainer['sampled_data'].shape[0])
        random_instance_vec = explainer['sampled_data'][random_idx, :]
        random_instance = [
            i + 1 for i, v in enumerate(random_instance_vec) if v == 0
        ]

        with explain_out:
            explain_out.clear_output(wait=True)
            plot_image_explanation(explainer, explain_class, random_instance)

            plt.show()

        # if explanation:
        #     pass
        # else:
        #     d = {'error': 'No explanation available'}
        # return d

    granularity_selection_slider = widgets.SelectionSlider(
        options=list(granularity_selection.keys()),
        value='low',
        description='Segmentation granularity:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True
    )
    colour_selection_toggle = widgets.ToggleButtons(
        options=list(colour_selection.keys()),
        description='Occlusion colour:',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        # tooltips=['Description of slow', 'Description of regular',
        #           'Description of fast'],
        # icons=['check'] * 3
    )
    class_selection_toggle = widgets.ToggleButtons(
        options=list(class_selection.keys()),
        description='Explained class:',
        disabled=False,
        button_style='',
    )
    explain_button = widgets.Button(
        description='Explain!',
        disabled=False,
        button_style='info',
        tooltip='Explain',
        icon='check'
    )
    explain_out = widgets.Output()

    explain_button.on_click(explain_action)
    explain_button._click_handlers(explain_button)  # pre-click the button

    surrogate_image_explainer = widgets.VBox([
        granularity_selection_slider,
        colour_selection_toggle,
        class_selection_toggle,
        explain_button,
        explain_out
    ])
    return surrogate_image_explainer


def build_tabular_blimey(instance,
                         class_to_explain,
                         data,
                         data_classes,
                         prediction_fn,
                         discretisation,
                         samples_number=50,
                         sample=True,
                         random_seed=42):
    """Builds a bLIMEy surrogate tabular explainer."""
    assert len(discretisation.keys()) == 2

    if sample:
        fatf.setup_random_seed(random_seed)
        augmenter = fatf_data_augmentation.Mixup(
            data, ground_truth=data_classes)
        sampled_data = augmenter.sample(
            instance, samples_number=samples_number)
    else:
        sampled_data = data.copy()
    sampled_data_probabilities = prediction_fn(sampled_data)

    if sample:
        sampled_data_classes = np.argmax(sampled_data_probabilities, axis=1)
    else:
        sampled_data_classes = data_classes.copy()

    # digitize data
    data_discretised = np.vstack([
        np.digitize(sampled_data[:, i], thresholds)
        for i, thresholds in discretisation.items()
    ]).T

    # digitize point
    point_discretised = np.array([
        np.digitize(instance[i], thresholds)
        for i, thresholds in discretisation.items()
    ])

    data_binarised = (data_discretised == point_discretised).astype(np.int8)

    # Train surrogate ridge
    surrogate = sklearn.linear_model.Ridge()
    surrogate.fit(
        data_binarised, sampled_data_probabilities[:, class_to_explain])

    explainers = dict(
        sampled_data=sampled_data,
        sampled_data_classes=sampled_data_classes,
        explanation=surrogate.coef_
    )
    return explainers


def plot_tabular_explanation(instance, instance_label, explained_label,
                             blimey, ranges, feature_map):
    """
    Plots a visualisation of the explained instance and a bar-plot explanation.
    """
    X = blimey['sampled_data']
    Y = blimey['sampled_data_classes']

    explained_features = sorted(list(ranges.keys()))
    assert len(explained_features) == 2
    x_ind, y_ind = explained_features[0], explained_features[1]
    x_name, x_range = feature_map[x_ind], ranges[x_ind]
    y_name, y_range = feature_map[y_ind], ranges[y_ind]

    importances = blimey['explanation']

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_alpha(0)
    fig.suptitle('Explained instance: {}    |    Explained class: {}'.format(
        instance_label, explained_label), fontsize=18)

    # plot /petal length (cm)/ vs /petal width/
    # x_name, y_name = 'petal length (cm)', 'petal width (cm)'
    # x_ind, y_ind = iris.feature_names.index(x_name), \
    #                iris.feature_names.index(y_name)
    x_min, x_max = X[:, x_ind].min() - .5, X[:, x_ind].max() + .5
    y_min, y_max = X[:, y_ind].min() - .5, X[:, y_ind].max() + .5
    #
    ax_l.scatter(X[:, x_ind], X[:, y_ind],
                 c=Y, cmap=plt.cm.Set1, edgecolor='k')
    ax_l.set_xlabel(x_name, fontsize=18)
    ax_l.set_ylabel(y_name, fontsize=18)
    #
    ax_l.set_xlim(x_min, x_max)
    ax_l.set_ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    ax_l.scatter(instance[x_ind], instance[y_ind],
                 c='yellow', marker='*', s=500, edgecolor='k')
    ax_l.vlines(x_range, -1, 10, linewidth=3)
    ax_l.hlines(y_range, -1, 10, linewidth=3)
    #
    ax_l.tick_params(axis='x', labelsize=18)
    ax_l.tick_params(axis='y', labelsize=18)


    x_dig_ = np.digitize(instance[x_ind], x_range)
    x_dig_list_ = ['-inf'] + [str(i) for i in x_range] + ['+inf']
    y_dig_ = np.digitize(instance[y_ind], y_range)
    y_dig_list_ = ['-inf'] + [str(i) for i in y_range] + ['+inf']
    x = [
        '{}\n{} < ... <= {}'.format(
            x_name,
            x_dig_list_[x_dig_],
            x_dig_list_[x_dig_ + 1]),
        '{}\n{} < ... <= {}'.format(
            y_name,
            y_dig_list_[y_dig_],
            y_dig_list_[y_dig_ + 1])
    ]
    #
    y = [abs(i) for i in importances]
    c = ['green' if i >= 0 else 'red' for i in importances]

    ax_r.set_xlim([0, 1.20])
    ax_r.set_ylim([-.5, len(x) - .5])
    ax_r.grid(False, axis='y')
    ax_r.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    ax_r.barh(x, y, height=.5, color=c)
    ax_r.set_yticklabels([])
    for i, v in enumerate(y):
        ax_r.text(v + .02, i + .15, '{:.4f}'.format(v),
                  fontweight='bold', fontsize=18)
        ax_r.text(v + .02, i - .2, x[i],
                  fontweight='bold', fontsize=18)

    # highlight explained spot
    x_dig_list_val_ = [0] + [i for i in x_range] + [8]
    ax_l.axvspan(x_dig_list_val_[x_dig_], x_dig_list_val_[x_dig_ + 1],
            facecolor='blue', alpha=0.2)
    y_dig_list_val_ = [-.5] + [i for i in y_range] + [3.5]
    ax_l.axhspan(y_dig_list_val_[y_dig_], y_dig_list_val_[y_dig_ + 1],
            facecolor='yellow', alpha=0.2)
    
    ax_r.tick_params(axis='x', labelsize=18)
    # ax_r.tick_params(axis='y', labelsize=18)

    plt.show()


def generate_tabular_widget(
        instances_to_explain,
        discretisation,
        class_map,
        feature_map,
        data,
        data_classes,
        prediction_fn,
        samples_number=50,
        sample=True,
        random_seed=42):
    """Creates an interactive bLIMEy tabular explanation widget."""

    def explain_action(obj):
        instance = instances_to_explain[lime_instance_toggle.value]
        instance_label = lime_instance_toggle.value
        label_to_explain = lime_class_toggle.value
        class_to_explain = class_map[lime_class_toggle.value]
        ranges = {x_ind: x_slider.value, y_ind: y_slider.value}

        blimey = build_tabular_blimey(
            instance, class_to_explain, data, data_classes, prediction_fn,
            ranges, samples_number, sample, random_seed)

        with lime_explain_out:
            lime_explain_out.clear_output(wait=True)
            plot_tabular_explanation(
                instance, instance_label, label_to_explain,
                blimey, ranges, feature_map)
            plt.show()

        # if explanation:
        #     pass
        # else:
        #     d = {'error': 'No explanation available'}
        # return d

    # Select 1 of three points
    lime_instance_toggle = widgets.ToggleButtons(
        options=list(instances_to_explain.keys()),
        description='Instance:',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        # tooltips=['Description of slow', 'Description of regular',
        #           'Description of fast'],
        # icons=['check'] * 3
    )
    # Select a class to explain
    lime_class_toggle = widgets.ToggleButtons(
        options=list(class_map.keys()),
        description='Class:',
        disabled=False,
        button_style='',
    )

    explained_features = sorted(list(discretisation.keys()))
    assert len(explained_features) == 2
    x_ind, y_ind = explained_features[0], explained_features[1]
    x_range = discretisation[x_ind]['range']
    x_step = discretisation[x_ind]['step']
    x_default = discretisation[x_ind]['default']
    x_name = feature_map[x_ind]
    if '.' in str(x_step):
        x_precision = len(str(x_step).split('.')[1])
    else:
        x_precision = len(str(x_step))
    y_range = discretisation[y_ind]['range']
    y_step = discretisation[y_ind]['step']
    y_default = discretisation[y_ind]['default']
    y_name = feature_map[y_ind]
    if '.' in str(y_step):
        y_precision = len(str(y_step).split('.')[1])
    else:
        y_precision = len(str(y_step))

    # Select two thresholds for the segmentation
    x_slider = widgets.FloatRangeSlider(
        value=x_default,
        min=min(x_range),
        max=max(x_range),
        step=x_step,
        description='X - {}:'.format(x_name),
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.{}f'.format(x_precision),
    )
    y_slider = widgets.FloatRangeSlider(
        value=y_default,
        min=min(y_range),
        max=max(y_range),
        step=y_step,
        description='Y - {}:'.format(y_name),
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.{}f'.format(y_precision),
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

    surrogate_tabular_explainer = widgets.VBox([
        lime_instance_toggle, lime_class_toggle, x_slider, y_slider,
        lime_explain_button, lime_explain_out
    ])
    return surrogate_tabular_explainer
