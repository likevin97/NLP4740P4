import json
import nltk
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score

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


def similarity(context_vector, question_vector):
    #FIND THE SIMILARITY SOMEHOW
    #returns the similarity
    pass


def main():
    corpus = createCorpus("training_sample.json")

    file = open("training_sample.json")
    j = json.load(file)

    predictions = {} #id:value

    similarity = {}

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