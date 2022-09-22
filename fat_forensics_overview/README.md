[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fat-forensics/resources/master?filepath=fat_forensics_overview)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fat-forensics/resources/blob/master/)
[![new BSD](https://img.shields.io/github/license/fat-forensics/resources.svg)](https://github.com/fat-forensics/resources/blob/master/LICENCE)  
[![Software Impacts](https://img.shields.io/badge/Software%20Impacts-10.1016/j.simpa.2022.100406-orange.svg)][simpa]

# FAT Forensics: A Python Toolbox for Algorithmic Fairness, Accountability and Transparency #

This directory contains a Jupyter Notebook that can be used to reproduce the
results presented in the "*FAT Forensics: A Python Toolbox for Algorithmic
Fairness, Accountability and Transparency*" paper.

The manuscript is published with [Software Impacts][simpa].

To run the notebook (`FAT_Forensics.ipynb`) you need to install a collection of
Python dependencies listed in the `requirements.txt` file (included in this
directory) by executing `pip install -r requirements.txt`.
Alternatively, you can run it via Binder or Colab by clicking the buttons
included above.

## Abstract ##

> Today, artificial intelligence systems driven by machine learning algorithms
> can be in a position to take important, and sometimes legally binding,
> decisions about our everyday lives.
> In many cases, however, these systems and their actions are neither
> regulated nor certified.
> To help counter the potential harm that such algorithms can cause we
> developed an open source toolbox that can analyse selected fairness,
> accountability and transparency aspects of the machine learning process:
> data (and their features), models and predictions, allowing to automatically
> and objectively report them to relevant stakeholders.
> In this paper we describe the design, scope, usage and impact of this
> Python package, which is published under the 3-Clause BSD
> open source licence.

## BibTeX ##
```
@article{sokol2022fat,
  title={{FAT Forensics}: {A} {Python} Toolbox for Algorithmic Fairness,
         Accountability and Transparency},
  author={Sokol, Kacper and Santos-Rodriguez, Raul and Flach, Peter},
  journal={Software Impacts},
  pages={100406},
  year={2022},
  publisher={Elsevier}
}
```

[simpa]: https://doi.org/10.1016/j.simpa.2022.100406
