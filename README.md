# voikko-sklearn

Python modules that extracts features from text documents for machine learning. Initially this is mostly for Finnish but may also
work for other languages supported by Voikko.

## Design goals

* The modules should have similar interface as those in
[sklearn.feature_extraction.text](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)
so that they can be used as drop in replacements when processing text in languages that are better supported by Voikko.
* The modules may depend on sklearn when they extend its functionality but they must also be usable with other machine learning libraries.
* Allow experiments and temporary workarounds for things that cannot be fixed in libvoikko due to API compatibility requirements.
  * For example case sensitive analyzer
* Easy to use API for developers and data scientist who do not know (and do not want to learn) all the options libvoikko provides
to tweak the analyzer.
* Provide working example code for those who want to do similar things (machine learning from Finnish text) in other programming languages.

## Non-goals

* Stable releases and stable API (for now). Basic functionality needs to be tested in real world first.
* Issues that can be fixed at their root (in libvoikko or voikko-fi) should be fixed there instead of working them around here.

## Requirements

* libvoikko 4.3 or later
* ```pip3 install scikit-learn```

### My OS does not have sufficiently new version of libvoikko, what to do?

#### On Linux

On Debian based systems including Ubuntu you can use the following commands to install development snapshot of latest libvoikko:
```
$ mkdir /tmp/voikko
$ cd /tmp/voikko
$ git clone https://github.com/voikko/debian-packages
$ debian-packages/tools/makevoikkodeb libvoikko
[The script will complain if some required dependencies are missing.
If that happens install the required packages and try the command again.]
$ sudo dpkg -i *.deb
```
The same procedure can be used for voikko-fi (just replace libvoikko with voikko-fi above).

Both libvoikko and voikko-fi are carefully developed and master branch is throughly tested so using it should work fine
for development work. The makevoikkodeb script also allows you to specify a release tag or commit id in case you want to build
a specific version instead of latest master.

#### On macOS ####

Using Homebrew
```
brew install libvoikko
pip3 install libvoikko
```
you can install both libvoikko and Finnish dictionary (voikko-fi). Nothing else is needed.

#### On Windows

Find the latest version of libvoikko-1.dll (32 or 64 bit according to your Python interpreter version) from
https://www.puimula.org/htp/testing/voikko-sdk/win-crossbuild/ and copy it somewhere in your $PATH.
Then ```pip3 install libvoikko```

In case you run into errors there are more detailed instructions here: https://voikko.puimula.org/python.html

