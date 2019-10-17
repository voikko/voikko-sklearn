# voikko-sklearn

Python modules that extract features from text documents for machine learning.

*Note: this is work in progress. The code is not usable yet.*

## Design goals

* The modules should have similar interface as those in
[sklearn.feature_extraction.text](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)
so that they can be used as drop in replacements when processing text in languages that are better supported by Voikko.
* The modules will not depend on sklearn so they can also be used with other machine learning libraries.

## Non-goals

* Stable releases and stable API (for now). Basic functionality needs to be tested in real world first.
* Issues that can be fixed at their root (in libvoikko or voikko-fi) should be fixed there instead of working them around here.
It is however OK to temporarily work around things that cannot be fixed in libvoikko due to API compatibility requirements.
