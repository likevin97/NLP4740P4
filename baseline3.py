import json
import nltk
from collections import Counter

# Simplified question
def getSimpleQuestion(question):
    text = nltk.word_tokenize(question)
    words = nltk.pos_tag(text)
    less_words = [wt for (wt, tag) in words if tag not in ["CC","DT","EX","IN","LS","POS","TO",".","\\",",",":","(",")"]]
    return less_words

def countWordsInParagraph(context):
    wordList = nltk.word_tokenize(context)
    counts = Counter(wordList)
    return counts

def uniqueWordsInDataset(prev_set, new_words):
	words = set(nltk.word_tokenize(new_words))
	new_set = prev_set.union(words)
	return new_set

def createEmptyVector(filename):
	file = open(filename)
	j = json.load(file)
	unique_words = set()

	data_length = len(j[u'data']) #442
	for data in range(data_length):
		paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

		for paragraph in range(paragraph_length):
			context = j[u'data'][data][u'paragraphs'][paragraph][u'context']
			question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

			unique_words = uniqueWordsInDataset(unique_words, context)

			for q in range(question_length):
				question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"]
				
				unique_words = uniqueWordsInDataset(unique_words, question)

	return unique_words

def similarity(context_vector, question_vector):
	#FIND THE SIMILARITY SOMEHOW
	#returns the similarity
	pass
def main():
	file = open("training.json")
	j = json.load(file)

	predictions = {} #id:value
	unique_words = createEmptyVector("training.json")
	vec_length = len(unique_words) #get length of set
	index = {}
	count = 0
	for word in unique_words:
			index[word] = count
			count += 1

	emptyVec = [0 for i in range(vec_length)]


	data_length = len(j[u'data']) #442
	for data in range(data_length):
		paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

		for paragraph in range(paragraph_length):
			context = j[u'data'][data][u'paragraphs'][paragraph][u'context']
			question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

			vector_context = emptyVec
			for word in context:
				vector_context[index[word]] += 1

			for q in range(question_length):
				question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"]
				ids = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["id"]

				#only training has is_impossible
				#is_impossible = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["is_impossible"]
				vector_question = emptyVec
				for word in question:
					vector_question[index[word]] += 1



				print (vector_question)
				print ("-----")
				print (vector_context)
				print ("-----")
				print (index)
				break
			break
		break
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
	with open('output3.json', 'w') as outfile:  
    	json.dump(predictions, outfile)

main()