[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fat-forensics/resources/master?filepath=surrogates_overview)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fat-forensics/resources/blob/master/)
[![new BSD](https://img.shields.io/github/license/fat-forensics/resources.svg)](https://github.com/fat-forensics/resources/blob/master/LICENCE)

# Interactive Overview of Tabular and Image Surrogates #

This directory contains a Jupyter Notebook that was presented during
the following events:

* [2021 BIAS Summer School][2021_bias-summer-school]; and
* [2021 TAILOR Summer School][2021_tailor-summer-school].

To run the notebook (`surrogates_overview.ipynb`) you need to install a
collection of Python dependencies listed in the `requirements.txt` file
(included in this directory) by executing `pip install -r requirements.txt`.
Alternatively, you can run it via Binder or Colab by clicking the buttons
listed above.

The `scripts` directory contains a collection of Python modules to help with
building and visualising interactive surrogate explainers of image and tabular
data.
It also implements a simple image classifier based on [PyTorch][pytorch] and
a number of interactive iPyWidgets for no-code experiments with these
explainers.

---

This Jupyter Notebook can also be rendered as a [reveal.js][reveal] slide show
through [RISE][rise].
To this end, it needs to be run from within a Jupyter Notebook environment
(not Jupyter Lab).
Next:

1. execute all cells; and
2. launch RISE presentation by clicking the bar chart icon
   (<img src="img/barchart.svg" width=20px />) shown in the
   Jupyter Notebook toolbar.

[2021_tailor-summer-school]: https://events.fat-forensics.org/2021_tailor-summer-school/
[2021_bias-summer-school]: https://events.fat-forensics.org/2021_bias/
[reveal]: https://revealjs.com/
[rise]: https://rise.readthedocs.io/
[pytorch]: https://pytorch.org/
