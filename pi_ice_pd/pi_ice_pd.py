"""
Permutation Importance, Individual Conditional Expectation & Partial Dependence
===============================================================================

This module implements a collection of helper functions for an interactive
builder of permutation importance, individual conditional expectation and
partial dependence explainers of tabular data based on iPyWidgets.

See <https://github.com/fat-forensics/resources/tree/master/pi_ice_pd>
for more details.
"""
# Author: Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import sklearn.inspection

import fatf.transparency.models.feature_influence as fatf_feature_influence
import fatf.vis.feature_influence as fatf_feature_influence_vis
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn')


def build_permutation_importance(
        data,
        data_labels,
        feature_names,
        model,
        metrics,
        repeats=100,
        random_seed=42
    ):
    """Calculates permutation feature importance."""
    pi_results = {}
    for metric in metrics:
        pi = sklearn.inspection.permutation_importance(
            model,
            data,
            data_labels,
            n_repeats=repeats,
            scoring=metric,
            random_state=random_seed)

        pi_results[metric] = []
        for feature_id, feature_name in enumerate(feature_names):
            pi_results[metric].append((
                feature_name,
                pi.importances_mean[feature_id],
                pi.importances_std[feature_id]
            ))

            # for i in pi.importances_mean.argsort()[::-1]:
            #     if pi.importances_mean[i] - 2 * pi.importances_std[i] > 0:
            #         print(f'{feature_name:<8}'
            #               f'{pi.importances_mean[feature_id]:.3f}'
            #               f' +/- {pi.importances_std[feature_id]:.3f}')

    return pi_results


def plot_permutation_importance(
        feature_importance,
        data,
        data_labels,
        feature_names,
        index_grouping,
        title
    ):
    """Plots a permutation importance explanation."""
    fig, (ax_l, ax_m, ax_r) = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_alpha(0)
    plt.suptitle(title, fontsize=18)

    # plot /petal length (cm)/ vs /petal width/
    x_ind, y_ind = index_grouping[0]
    x_name, y_name = feature_names[x_ind], feature_names[y_ind]
    x_min, x_max = data[:, x_ind].min() - .5, data[:, x_ind].max() + .5
    y_min, y_max = data[:, y_ind].min() - .5, data[:, y_ind].max() + .5
    #
    ax_l.scatter(data[:, x_ind], data[:, y_ind],
                 c=data_labels, cmap=plt.cm.Set1, edgecolor='k')
    ax_l.set_xlabel(x_name, fontsize=18)
    ax_l.set_ylabel(y_name, fontsize=18)
    #
    ax_l.set_xlim(x_min, x_max)
    ax_l.set_ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    ax_l.tick_params(axis='x', labelsize=18)
    ax_l.tick_params(axis='y', labelsize=18)

    # plot /sepal length (cm)/ vs /sepal width/
    x_ind, y_ind = index_grouping[1]
    x_name, y_name = feature_names[x_ind], feature_names[y_ind]
    x_min, x_max = data[:, x_ind].min() - .5, data[:, x_ind].max() + .5
    y_min, y_max = data[:, y_ind].min() - .5, data[:, y_ind].max() + .5
    #
    ax_m.scatter(data[:, x_ind], data[:, y_ind],
                 c=data_labels, cmap=plt.cm.Set1, edgecolor='k')
    ax_m.set_xlabel(x_name, fontsize=18)
    ax_m.set_ylabel(y_name, fontsize=18)
    #
    ax_m.set_xlim(x_min, x_max)
    ax_m.set_ylim(y_min, y_max)
    ax_m.tick_params(axis='x', labelsize=18)
    ax_m.tick_params(axis='y', labelsize=18)

    # plot explanation
    x = ['\n'.join(i[0].split()) for i in feature_importance]
    y = [i[1] for i in feature_importance]
    e = [i[2] for i in feature_importance]
    ax_r.bar(x, y,
             yerr=e, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax_r.set_ylim(0, 1.5)
    ax_r.tick_params(axis='x', labelsize=18)
    ax_r.tick_params(axis='y', labelsize=18)

    plt.show()


def generate_pi_widget(
        metrics,
        permutation_importance,
        index_grouping,
        data,
        data_labels,
        feature_names
    ):
    """Builds iPyWidget interactive feature importance explainer."""

    def explain_pfi(obj):
        metric_type = metrics[pfi_metric_toggle.value]
        pfi = permutation_importance[metric_type]

        with pfi_explain_out:
            pfi_explain_out.clear_output(wait=True)
            plot_permutation_importance(
                pfi,
                data,
                data_labels,
                feature_names,
                index_grouping,
                pfi_metric_toggle.value
            )
            plt.show()

    pfi_metric_toggle = widgets.ToggleButtons(
        options=list(metrics.keys()),
        description='Metric:',
        disabled=False,
        button_style=''
    )
    pfi_explain_button = widgets.Button(
        description='Explain!',
        disabled=False,
        button_style='info',
        tooltip='Explain',
        icon='check'
    )
    pfi_explain_out = widgets.Output()

    pfi_explain_button.on_click(explain_pfi)
    # pre-click the button
    pfi_explain_button._click_handlers(pfi_explain_button)

    pfi_widget = widgets.VBox([
        pfi_metric_toggle, pfi_explain_button, pfi_explain_out])
    return pfi_widget


def build_ice_pd(
        data,
        model,
        feature_indices,
        samples_no=100
    ):
    """Calculates individual conditional expectation and partial dependence."""
    ice_pd = {}
    for i in feature_indices:
        ice_array, linspace = \
            fatf_feature_influence.individual_conditional_expectation(
                data, model, i, steps_number=samples_no)
        pd_array = fatf_feature_influence.partial_dependence_ice(ice_array)
        ice_pd[i] = dict(ice=ice_array, pd=pd_array, linspace=linspace)
    return ice_pd


def plot_ice_pd(
        ice_pd,
        explanation_class,
        class_labels,
        feature_names,
        data_point=None,
        data_point_class=None,
        model=None,
        discretisation=None  # bLIMEy discretisation
    ):
    """Plots individual conditional expectation and partial dependence."""
    colours = ['blue', 'yellow']
    explanation_class_name = class_labels[explanation_class]
    exp_ = sorted(list(ice_pd.keys()))
    assert len(exp_) == 2, 'Plotting works only with exactly 2 features.'

    fig, (ax_u, ax_d) = plt.subplots(2, 1, figsize=(12, 7))
    fig.patch.set_alpha(0)
    if data_point is None:
        title = 'Explained class: {}'.format(explanation_class_name)
    else:
        assert data_point_class is not None, 'Class must be provided as well.'
        explained_point_pred = model.predict_proba(
            [data_point])[0][explanation_class]
        title = 'Explained instance: {}    |    Explained class: {}'.format(
            class_labels[data_point_class], explanation_class_name)
    fig.suptitle(title, fontsize=18)

    for i, ax in enumerate([ax_u, ax_d]):
        x_ = ice_pd[exp_[i]]
        linspace_x, ice_x, pd_x = x_['linspace'], x_['ice'], x_['pd']
        feature_name = feature_names[exp_[i]]

        ax.set_xlim([linspace_x[0], linspace_x[-1]])
        ax.set_ylim([-0.05, 1.05])
        _ = fatf_feature_influence_vis.plot_individual_conditional_expectation(
            ice_x,
            linspace_x,
            explanation_class,
            class_name=explanation_class_name,
            feature_name=feature_name,
            plot_axis=ax)
        # ice_plot_2_figure, ice_plot_2_axis = ice_plot_2
        _ = fatf_feature_influence_vis.plot_partial_dependence(
            pd_x,
            linspace_x,
            explanation_class,
            class_name=explanation_class_name,
            feature_name=feature_name,
            plot_axis=ax)
        ax.set_ylabel(explanation_class_name, fontsize=18)
        ax.set_title('')

        if discretisation is not None:
            ax.vlines(discretisation[exp_[i]], -.05, 1.05, linewidth=3)
        if data_point is not None:
            ax.scatter(data_point[exp_[i]], explained_point_pred,
                       c='yellow', marker='*', s=500, edgecolor='k')
        if discretisation is not None and data_point is not None:
            x_dig_ = np.digitize(data_point[exp_[i]], discretisation[exp_[i]])
            x_dig_list_val_ = (
                [linspace_x[0] - 1]
                + [i for i in discretisation[exp_[i]]]
                + [linspace_x[-1] + 1]
            )
            ax.axvspan(x_dig_list_val_[x_dig_], x_dig_list_val_[x_dig_ + 1],
                    facecolor=colours[i], alpha=0.1)

        ax.xaxis.label.set_fontsize(18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

    plt.show()


def generate_ice_pd_widget(
        ice_pd,
        class_labels,
        feature_names,
        instances_to_explain=None,
        model=None,
        show_discretisation=False
    ):
    """
    Builds interactive iPyWidget explainer for individual conditional
    expectation and partial dependence.
    """

    def explain_ice_pd(obj):
        if instance_toggle is not None:
            instance_class_ = instance_toggle.value
            instance_class_id_ = class_map[instance_class_]
            instance_ = instances_to_explain[instance_class_]
        else:
            instance_class_ = None
            instance_class_id_ = None
            instance_ = None

        explained_class_ = class_toggle.value
        explained_class_id_ = class_map[explained_class_]

        if show_discretisation:
            discretisation = {
                exp_[0]: x_slider.value,
                exp_[1]: y_slider.value
            }
        else:
            discretisation = None

        with icepd_explain_out:
            icepd_explain_out.clear_output(wait=True)
            plot_ice_pd(
                ice_pd,
                explained_class_id_,
                class_labels,
                feature_names,
                data_point=instance_,
                data_point_class=instance_class_id_,
                model=model,
                discretisation=discretisation  # bLIMEy discretisation
            )
            plt.show()

    exp_ = sorted(list(ice_pd.keys()))
    assert len(exp_) == 2, 'Widgets work only with exactly 2 features.'
    class_map = {j: i for i, j in class_labels.items()}
    widgets_ = []

    if instances_to_explain is not None:
        # Select 1 of three points
        instance_toggle = widgets.ToggleButtons(
            options=list(instances_to_explain.keys()),
            description='Instance:',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            # tooltips=['Description of slow', 'Description of regular',
            #           'Description of fast'],
            # icons=['check'] * 3
        )
        widgets_.append(instance_toggle)
        assert model is not None, 'Model is required with an instance.'
    else:
        instance_toggle = None

    # Select a class to explain
    class_toggle = widgets.ToggleButtons(
        options=list(class_map.keys()),
        description='Class:',
        disabled=False,
        button_style='',
    )
    widgets_.append(class_toggle)

    if show_discretisation:
        x_ = ice_pd[exp_[0]]
        x_step = x_['linspace'][1] - x_['linspace'][0]
        if '.' in str(x_step):
            x_precision = len(str(x_step).strip('0').split('.')[1])
        else:
            x_precision = len(str(x_step))
        x_precision = 2 if x_precision > 2 else x_precision
        # Select two thresholds for the segmentation
        x_slider = widgets.FloatRangeSlider(
            value=(x_['linspace'][0], x_['linspace'][-1]),
            min=x_['linspace'][0],
            max=x_['linspace'][-1],
            step=x_step,
            description='[X] {}:'.format(feature_names[exp_[0]]),
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.{}f'.format(x_precision),
        )
        x_ = ice_pd[exp_[1]]
        x_step = x_['linspace'][1] - x_['linspace'][0]
        if '.' in str(x_step):
            x_precision = len(str(x_step).split('.')[1])
        else:
            x_precision = len(str(x_step))
        x_precision = 2 if x_precision > 2 else x_precision
        y_slider = widgets.FloatRangeSlider(
            value=(x_['linspace'][0], x_['linspace'][-1]),
            min=x_['linspace'][0],
            max=x_['linspace'][-1],
            step=x_step,
            description='[Y] {}:'.format(feature_names[exp_[1]]),
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.{}f'.format(x_precision),
        )
        widgets_.append(x_slider)
        widgets_.append(y_slider)
    else:
        x_slider = None
        y_slider = None

    icepd_explain_button = widgets.Button(
        description='Explain!',
        disabled=False,
        button_style='info',
        tooltip='Explain',
        icon='check'
    )
    widgets_.append(icepd_explain_button)

    icepd_explain_out = widgets.Output()
    widgets_.append(icepd_explain_out)

    icepd_explain_button.on_click(explain_ice_pd)
    # pre-click the button
    icepd_explain_button._click_handlers(icepd_explain_button)

    ice_pd_widget = widgets.VBox(widgets_)
    return ice_pd_widget
