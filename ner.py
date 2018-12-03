import json
import nltk
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

# Simplified question
def getPOS(corpus):
    pos_tag = []
    for sentence in corpus:
        text = nltk.word_tokenize(sentence)
        words = nltk.pos_tag(text)
        for word in words:
            pos_tag.append(word)

    #less_words = [wt for (wt, tag) in words if tag not in ["CC","DT","EX","IN","LS","POS","TO",".","\\",",",":","(",")"]]
    #return less_words
    return pos_tag

def countWordsInParagraph(context):
    wordList = nltk.word_tokenize(context)
    counts = Counter(wordList)
    return counts

def createCorpus(stopwords, filename):
    file = open(filename)
    j = json.load(file)

    corpus = []

    data_length = len(j[u'data']) #442
    for data in range(data_length):
        paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

        for paragraph in range(paragraph_length):
            context = nltk.word_tokenize(j[u'data'][data][u'paragraphs'][paragraph][u'context'].lower())

            corpus.append(" ".join([w for w in context if not w in stopwords]))
            #corpus.append(context)

            question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

            for q in range(question_length):
                question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"].lower()

                corpus.append(question)
    file.close()
    return corpus

def ourLemmatize(dictonary, corpus):
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
    stop_words = set(stopwords.words('english'))
    stop_words.union(set(string.punctuation))
    corpus = createCorpus(stop_words, "training_sample.json")
    #print (corpus)
    #print ("=---=-------")

    file = open("training_sample.json")
    j = json.load(file)

    predictions = {} #id:value

    similarity = {}

    pos = getPOS(corpus)
    pos_dict = {}
    for (word, tag) in pos:
        wtag = tag[0].lower()
        wtag = wtag if wtag in ["a","r","n","v"] else None
        pos_dict[word] = wtag


    corpus = ourLemmatize(pos_dict, corpus)

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    vectorsArrayForm = vectors.toarray()

    counter = 0
    question_words = ["who", "what", "when", "where", "why"]

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

                #similarity[question_id] = jaccard_similarity_score(context_vector, question_vector)

                question = corpus[counter]

                question_array = [question]

                question_pos = getPOS(question_array)

                print (question_pos)
                q_word = ""
                q_verb = []
                for (w,t) in question_pos:
                    if w in question_words:
                        q_word = w
                    if t.startswith("V"):
                        q_verb.append(w)
                
                print ("Q-Word: " + q_word)
                print ("Q-Verb: " + str(q_verb))

                counter += 1
            break
        break




    # print(similarity)

main()