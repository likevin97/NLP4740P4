import json
import nltk
from collections import Counter

def main():
	file = open("training.json")
	j = json.load(file)

	predictions = {} #id:value

	data_length = len(j[u'data']) #442
	for data in range(data_length):
		paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

		for paragraph in range(paragraph_length):
			context = j[u'data'][data][u'paragraphs'][paragraph][u'context']
			question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

			for q in range(question_length):
				question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"]
				ids = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["id"]
				predictions[ids] = 1

	with open('output1.json', 'w') as outfile:  
		json.dump(predictions, outfile)

main()