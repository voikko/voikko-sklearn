# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
# The Original Code is Libvoikko: Library of natural language processing tools.
# The Initial Developer of the Original Code is Harri Pitkänen <hatapitk@iki.fi>.
# Portions created by the Initial Developer are Copyright (C) 2019
# the Initial Developer. All Rights Reserved.
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.

# This library requires Python version 3.5 or newer.

from libvoikko import Voikko, Token
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy

class VoikkoCountVectorizer(CountVectorizer):

	FINNISH_STOPWORD_CLASSES = ["huudahdussana", "seikkasana", "lukusana", "asemosana", "sidesana", "suhdesana", "kieltosana"]

	def __init__(self, langtag="fi", binary=False, stop_word_classes=[]):
		self.voikko = Voikko(langtag)
		self.stop_word_classes = set(stop_word_classes)
		super().__init__(binary=binary)

	def terminate(self):
		self.voikko.terminate()

	def build_analyzer(self):
		check_stop_words = len(self.stop_word_classes) > 0
		def analyse_word(word):
			baseform = None
			is_stop_word = False
			for analysis in self.voikko.analyze(word):
				if check_stop_words and "CLASS" in analysis and analysis["CLASS"] in self.stop_word_classes:
					is_stop_word = True
				elif "BASEFORM" in analysis:
					new_baseform = analysis["BASEFORM"]
					if baseform is not None and baseform != new_baseform:
						return word.lower()
					baseform = new_baseform
				else:
					return word.lower()
			if baseform is None:
				if is_stop_word:
					return None
				return word.lower()
			return baseform
		def analyse_text(text):
			baseforms = [analyse_word(token.tokenText) for token in self.voikko.tokens(text) if token.tokenType == Token.WORD]
			if check_stop_words:
				return [baseform for baseform in baseforms if baseform is not None]
			return baseforms
		return analyse_text

class VoikkoAttributeVectorizer:
	"""Converts a collection of text documents to a matrix of counts of words
	having specific value for enumerated morphological analysis attributes.
	
	Examples
	--------
	>>> from voikko_sklearn import VoikkoAttributeVectorizer
	>>> corpus = [
	...     'Koiran karvat olivat takussa.',
	...     'Kissamme goli vanha.'
	... ]
	>>> vectorizer = VoikkoAttributeVectorizer(['NUMBER', 'PERSON'], langtag='fi')
	>>> print(vectorizer.get_feature_names())
	['unknown', 'NUMBER_plural', 'NUMBER_singular', 'PERSON_1', 'PERSON_2', 'PERSON_3', 'PERSON_4']
	>>> X = vectorizer.transform(corpus)
	>>> print(X.toarray())
	[[0.         0.5        0.5        0.         0.         0.25       0.        ]
	[0.33333333 0.         0.66666667 0.         0.         0.         0.        ]]
	"""
	
	def __init__(self, attributes, langtag="fi"):
		self.input = input
		self.attributes = attributes
		self.voikko = Voikko(langtag)
		self.__init_feature_names()

	def __init_feature_names(self):
		self.feature_names = ['unknown']
		self.feature_name_to_index = {'unknown' : 0}
		for attribute in self.attributes:
			values = self.voikko.attributeValues(attribute)
			if values is None:
				raise ValueError("Attribute '" + attribute + "' does not exist or is not categorial.")
			values.sort()
			for value in values:
				name = attribute + '_' + value
				self.feature_name_to_index[name] = len(self.feature_names)
				self.feature_names.append(name)

	def terminate(self):
		self.voikko.terminate()

	def build_tokenizer(self):
		return lambda text: [token.tokenText for token in self.voikko.tokens(text) if token.tokenType == Token.WORD]

	def get_feature_names(self):
		return self.feature_names

	def __transform_document(self, document, target_vector):
		words = self.build_tokenizer()(document)
		wordcount = len(words)
		if wordcount == 0:
			return
		for word in words:
			analysis_list = self.voikko.analyze(word)
			count = len(analysis_list)
			if count == 0:
				target_vector[0] += 1
			else:
				for analysis in analysis_list:
					for attribute in self.attributes:
						if attribute in analysis:
							value = analysis[attribute]
							target_vector[self.feature_name_to_index[attribute + "_" + value]] += 1.0 / count
		target_vector /= wordcount

	def transform(self, document_list):
		document_count = len(document_list)
		vector_length = len(self.feature_names)
		data = numpy.zeros((document_count, vector_length), dtype=numpy.float64)
		for i in range(document_count):
			self.__transform_document(document_list[i], data[i])
		return csr_matrix(data)

	def fit(self, document_list):
		return self

	def fit_transform(self, document_list):
		return self.transform(document_list)
