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

def main():
	file = open("development.json")
	j = json.load(file)

	predictions = {} #id:value

	data_length = len(j[u'data']) #442
	for data in range(data_length):
		paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

		for paragraph in range(paragraph_length):
			context = j[u'data'][data][u'paragraphs'][paragraph][u'context']
			question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

			dic = countWordsInParagraph(context)

			for q in range(question_length):
				question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"]
				ids = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["id"]

				#only training has is_impossible
				#is_impossible = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["is_impossible"]

				#deletes useless information from question
				new_question = getSimpleQuestion(question)
				total = 0
				for word in new_question:
					if word in dic:
						total+= dic[word]

				if total >= 1:
					predictions[ids] = 1
				else:
					predictions[ids] = 0
	with open('output2.json', 'w') as outfile:  
    	json.dump(predictions, outfile)

main()