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

import unittest
import numpy as np
from voikko_sklearn import VoikkoAttributeVectorizer, VoikkoCountVectorizer
from scipy.sparse import csr_matrix

class VoikkoAttributeVectorizerTest(unittest.TestCase):
	
	def test_init_and_terminate(self):
		v = VoikkoAttributeVectorizer([])
		v.terminate()

	def test_build_tokenizer(self):
		vectorizer = VoikkoAttributeVectorizer([])
		tokenizer = vectorizer.build_tokenizer()
		tokens = tokenizer('Kissa ei ole koira. Ei todellakaan ole.')
		self.assertEqual(['Kissa', 'ei', 'ole', 'koira', 'Ei', 'todellakaan', 'ole'], tokens)

	def test_get_feature_names(self):
		vectorizer = VoikkoAttributeVectorizer(['NUMBER', 'PERSON'])
		names = vectorizer.get_feature_names()
		expected = ['unknown', 'NUMBER_plural', 'NUMBER_singular', 'PERSON_1', 'PERSON_2', 'PERSON_3', 'PERSON_4']
		self.assertEqual(expected, names)

	def test_get_feature_names_unknown_feature_raises(self):
		self.assertRaises(ValueError, lambda: VoikkoAttributeVectorizer(['KISSA']))

	def test_get_feature_names_non_categorial_feature_raises(self):
		self.assertRaises(ValueError, lambda: VoikkoAttributeVectorizer(['BASEFORM']))

	def test_fit(self):
		vectorizer = VoikkoAttributeVectorizer(['NUMBER'])
		# fit should not do anything, just return self
		self.assertIs(vectorizer, vectorizer.fit(['Kissa ei ole koira.']))

	def test_transform(self):
		vectorizer = VoikkoAttributeVectorizer(['NUMBER'])
		X = vectorizer.transform(['Kissa ja jaf koira.'])
		self.assertIsInstance(X, csr_matrix)
		self.assertEqual((1, 3), X.shape)
		data = X.toarray()
		self.assertEqual(0.25, data[0][0]) # unknown
		self.assertEqual(0, data[0][1]) # plural
		self.assertEqual(0.5, data[0][2]) # singular

	def test_transform_two_empty_documents(self):
		vectorizer = VoikkoAttributeVectorizer(['NUMBER'])
		X = vectorizer.transform(['', '.'])
		self.assertIsInstance(X, csr_matrix)
		self.assertEqual((2, 3), X.shape)
		data = X.toarray()
		self.assertEqual(0, np.count_nonzero(data))

	def test_transform_ambiguity(self):
		vectorizer = VoikkoAttributeVectorizer(['MOOD'])
		X = vectorizer.transform(['tulen'])
		self.assertEqual((1, 10), X.shape)
		data = X.toarray()
		self.assertEqual(1, np.count_nonzero(data))
		self.assertEqual(0.5, data[0][8])

	def test_fit_transform(self):
		# should do the same as fit
		vectorizer = VoikkoAttributeVectorizer(['NUMBER'])
		X = vectorizer.fit_transform(['Kissa ja jaf koira.'])
		self.assertIsInstance(X, csr_matrix)


class VoikkoCountVectorizerTest(unittest.TestCase):

	def test_works_as_count_vectorizer_for_unknown_words(self):
		vectorizer = VoikkoCountVectorizer()
		X = vectorizer.fit_transform(['Allefsf nodfte nodfte blää.'])
		self.assertEqual(['allefsf', 'blää', 'nodfte'], vectorizer.get_feature_names())
		self.assertEqual((1, 3), X.shape)
		data = X.toarray()[0]
		self.assertEqual(1, data[0])
		self.assertEqual(1, data[1])
		self.assertEqual(2, data[2])

	def test_unambiguous_words_are_lemmatized(self):
		vectorizer = VoikkoCountVectorizer()
		X = vectorizer.fit_transform(['Tulen töihin, en elä töillä.'])
		self.assertEqual(['ei', 'elää', 'tulen', 'työ'], vectorizer.get_feature_names())
		self.assertEqual((1, 4), X.shape)
		data = X.toarray()[0]
		self.assertEqual(1, data[0])
		self.assertEqual(1, data[1])
		self.assertEqual(1, data[2])
		self.assertEqual(2, data[3])
		vectorizer.terminate()

	def test_binary(self):
		vectorizer = VoikkoCountVectorizer(binary=True)
		X = vectorizer.fit_transform(['Kissa, kissan.'])
		self.assertEqual(['kissa'], vectorizer.get_feature_names())
		self.assertEqual((1, 1), X.shape)
		data = X.toarray()[0]
		self.assertEqual(1, data[0])
		vectorizer.terminate()

if __name__ == "__main__":
	suites = [unittest.TestLoader().loadTestsFromTestCase(cls) for cls in [VoikkoAttributeVectorizerTest, VoikkoCountVectorizerTest]]
	unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(suites))
