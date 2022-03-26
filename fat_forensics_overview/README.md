[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fat-forensics/resources/master?filepath=fat_forensics_overview)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fat-forensics/resources/blob/master/)
[![new BSD](https://img.shields.io/github/license/fat-forensics/resources.svg)](https://github.com/fat-forensics/resources/blob/master/LICENCE)

# FAT Forensics: A Python Toolbox for Algorithmic Fairness, Accountability and Transparency #

This directory contains a Jupyter Notebook that can be used to reproduce the
results presented in the "*FAT Forensics: A Python Toolbox for Algorithmic
Fairness, Accountability and Transparency*" paper.

The manuscript is available on [arXiv][arXiv:1909.05167].

To run the notebook (`FAT_Forensics.ipynb`) you need to install a collection of
Python dependencies listed in the `requirements.txt` file (included in this
directory) by executing `pip install -r requirements.txt`.
Alternatively, you can run it via Binder or Colab by clicking the buttons
included above.

## Abstract ##

> Machine learning algorithms can take important, and sometimes legally binding,
> decisions about our everyday life.
> In many cases, however, these systems and their actions are neither regulated
> nor certified.
> Given the potential harm that such algorithms can cause, their fairness,
> accountability and transparency are of paramount importance.
> Recent literature suggested voluntary self-reporting on these aspects of
> predictive systems -- e.g., "datasheets for datasets" -- but their scope is
> often limited to a single component of a machine learning pipeline and their
> composition requires manual labour.
> To resolve this impasse and ensure high-quality, fair, transparent and
> reliable data-driven models, we developed an open source toolbox that can
> analyse selected fairness, accountability and transparency characteristics
> of these systems to automatically and objectively report them to relevant
> stakeholders.
> The software provides functionality for inspecting the aforementioned
> properties of all aspects of the machine learning process: data
> (and their features), models and predictions.
> In this paper we describe the design, scope and usage examples of this Python
> package, which is published under the BSD 3-Clause open source licence.

## BibTeX ##
```
@article{sokol2019fat,
  title={{FAT} {F}orensics: {A} {P}ython toolbox for algorithmic fairness,
         accountability and transparency},
  author={Sokol, Kacper and Santos-Rodriguez, Raul and Flach, Peter},
  journal={arXiv preprint arXiv:1909.05167},
  year={2019}
}
```

[arXiv:1909.05167]: https://arxiv.org/abs/1909.05167
