import json
import nltk
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
from scipy import spatial

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
            context = j[u'data'][data][u'paragraphs'][paragraph][u'context']

            corpus.append(context)

            question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

            for q in range(question_length):
                question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"]

                corpus.append(question)
    file.close()
    return corpus


def createCorpusBySentence(filename):
    file = open(filename)
    j = json.load(file)

    corpus = []

    data_length = len(j[u'data']) #442
    for data in range(data_length):
        paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

        for paragraph in range(paragraph_length):
            context = j[u'data'][data][u'paragraphs'][paragraph][u'context']

            context_sentences = context.split(".")

            for context_sentence in context_sentences:
                corpus.append(context_sentence)
            
            question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

            for q in range(question_length):
                question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"]

                corpus.append(question)

    file.close()
    return corpus


def similarity(file_name):
    corpus = createCorpus(file_name)

    file = open(file_name)
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
    
    return similarity


def similarityBySentence(file_name):
    corpus = createCorpusBySentence(file_name)

    file = open(file_name)
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

            context = j[u'data'][data][u'paragraphs'][paragraph][u'context']

            context_sentences = context.split(".")

            num_sentences = len(context_sentences)

            sentence_start = counter

            question_offset = 0


            question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])
            for q in range(question_length):
                question_id = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["id"]

                question_vector = vectorsArrayForm[sentence_start + question_offset + num_sentences]

                max_sim = 0

                for context_sentence in range(num_sentences):

                    context_sentence_vector = vectorsArrayForm[counter]

                    sentence_sim = 1 - spatial.distance.cosine(context_sentence_vector, question_vector)

                    if (sentence_sim > max_sim):
                        max_sim = sentence_sim

                    counter += 1
                
                similarity[question_id] = max_sim
                
                counter = sentence_start
                question_offset += 1
            
            counter = sentence_start + question_offset + num_sentences
            
    
    return similarity

def main():

    sim_dic = similarity("training.json")


    min_sim = 1.0
    max_sim = 0.0

    for key, value in sim_dic.items():
        if (value < min_sim):
            min_sim = value
        if (value > max_sim):
            max_sim = value
    

    print ("Minimum Jaccard Similarity: " + str(min_sim))
    print ("Maximum Jaccard Similarity: " + str(max_sim))

    # predictions = {}

    # for key, value in sim_dic.items():
    #     if value > 0.995:
    #         predictions[key] = 1
    #     else:
    #         predictions[key] = 0

    # with open('output3.json', 'w') as outfile:  
    #     json.dump(predictions, outfile)

main()