import json
import nltk
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
from nltk.stem import WordNetLemmatizer

# Simplified question
def getPOS(corpus):
	text = nltk.word_tokenize(corpus)
	words = nltk.pos_tag(text)
	#less_words = [wt for (wt, tag) in words if tag not in ["CC","DT","EX","IN","LS","POS","TO",".","\\",",",":","(",")"]]
	#return less_words
	return words

def countWordsInParagraph(context):
	wordList = nltk.word_tokenize(context)
	counts = Counter(wordList)
	return counts

def createCorpus(filename):
	file = open(filename)
	j = json.load(file)

	corpus = []

	data_length = len(j[u'data']) #442
	for data in range(data_length):
		paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

		for paragraph in range(paragraph_length):
			context = j[u'data'][data][u'paragraphs'][paragraph][u'context'].lower()

			corpus.append(context)

			question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

			for q in range(question_length):
				question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"].lower()

				corpus.append(question)
	file.close()
	return corpus

def lemmatize(dictonary, corpus):
	lemma = WordNetLemmatizer()
	corp = []
	for sentence in corpus:
		s = nltk.word_tokenize(sentence)
		sent = []
		for word in s:
			if dictonary[word] != None:
				nw = lemma.lemmatize(word, dictonary[word])
				sent.append(nw)
			else:
				sent.append(word)
		corp.append(" ".join(sent))
	return corp


def main():
	corpus = createCorpus("training_sample.json")
	#print (corpus)
	#print ("=---=-------")

	file = open("training_sample.json")
	j = json.load(file)

	predictions = {} #id:value

	similarity = {}

	pos = getPOS(" ".join(corpus))
	pos_dict = {}
	for (word, tag) in pos:
		# if tag.startswith("J"):
		# 	pos_dict[word] = wordnet.ADJ
		# elif tag.startswith("V"):
		# 	pos_dict[word] = wordnet.VERB
		# elif tag.startswith("N"):
		# 	pos_dict[word] = wordnet.NOUN
		# elif tag.startswith("R"):
		# 	pos_dict[word] = wordnet.ADV
		# else:
		# 	pos_dict[word] = ""

		wtag = tag[0].lower()
		wtag = wtag if wtag in ["a","r","n","v"] else None
		pos_dict[word] = wtag

	#print (pos_dict)

	#print ("household " + pos_dict["household"])

	corpus = lemmatize(pos_dict, corpus)

	#print (corpus)

	vectorizer = CountVectorizer()
	vectors = vectorizer.fit_transform(corpus)
	vectorsArrayForm = vectors.toarray()

	counter = 0

	data_length = len(j[u'data']) #442
	for data in range(data_length):

		paragraph_length = len(j[u'data'][data][u'paragraphs']) #66
		for paragraph in range(paragraph_length):

			context_vector = vectorsArrayForm[counter]

			counter += 1

			question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])
			for q in range(question_length):
				question_id = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["id"]

				question_vector = vectorsArrayForm[counter]

				similarity[question_id] = jaccard_similarity_score(context_vector, question_vector)

				counter += 1


	print(similarity)
	#s = similarity(vector_context, vector_quesetion)


	# #deletes useless information from question
	# new_question = getSimpleQuestion(question)
	# total = 0
	# for word in new_question:
	# 	if word in dic:
	# 		total+= dic[word]

	# if total >= 1:
	# 	predictions[ids] = 1
	# else:
	# 	predictions[ids] = 0
	# with open('output3.json', 'w') as outfile:  
	#     json.dump(predictions, outfile)

main()