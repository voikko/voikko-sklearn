# voikko-sklearn

Python modules that extracts features from text documents for machine learning. Initially this is mostly for Finnish but may also
work for other languages supported by Voikko.

*Note: this is work in progress. The code is not usable yet.*

## Design goals

* The modules should have similar interface as those in
[sklearn.feature_extraction.text](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)
so that they can be used as drop in replacements when processing text in languages that are better supported by Voikko.
* The modules will not depend on sklearn so they can also be used with other machine learning libraries.
* Allow experiments and temporary workarounds for things that cannot be fixed in libvoikko due to API compatibility requirements.
  * For example case sensitive analyzer
* Easy to use API for developers and data scientist who do not know (and do not want to learn) all the options libvoikko provides
to tweak the analyzer.
* Provide working example code for those who want to do similar things (machine learning from Finnish text) in other programming languages.

## Non-goals

* Stable releases and stable API (for now). Basic functionality needs to be tested in real world first.
* Issues that can be fixed at their root (in libvoikko or voikko-fi) should be fixed there instead of working them around here.

## Requirements

* Latest version of libvoikko (Git master may be required)
