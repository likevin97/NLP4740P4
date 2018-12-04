import json
import nltk
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from tqdm import tqdm

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
    for data in tqdm(range(data_length)):
        paragraph_length = len(j[u'data'][data][u'paragraphs']) #66

        for paragraph in range(paragraph_length):
            # context = nltk.word_tokenize(j[u'data'][data][u'paragraphs'][paragraph][u'context'])
            context = j[u'data'][data][u'paragraphs'][paragraph][u'context']

            # corpus.append(" ".join([w for w in context if not w in stopwords]))
            corpus.append(context)

            question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])

            for q in range(question_length):
                question = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["question"]

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
    # stop_words.union(set(string.punctuation))
    corpus = createCorpus(stop_words, "development.json")
    #print (corpus)
    #print ("=---=-------")

    file = open("development.json")
    j = json.load(file)

    predictions = {} #id:value

    similarity = {}

    # pos = getPOS(corpus)
    # pos_dict = {}
    # for (word, tag) in pos:
    #     wtag = tag[0].lower()
    #     wtag = wtag if wtag in ["a","r","n","v"] else None
    #     pos_dict[word] = wtag


    # corpus = ourLemmatize(pos_dict, corpus)


    counter = 0
    question_words = ["who", "what", "when", "where", "why", "how", "which", "whose", "whom", "is", "was", "are", "does", "did", "were", "can", "do", "has", "had", "name"]

    data_length = len(j[u'data']) #442
    for data in range(data_length):

        paragraph_length = len(j[u'data'][data][u'paragraphs']) #66
        for paragraph in range(paragraph_length):

            context = corpus[counter]

            counter += 1

            question_length = len(j[u'data'][data][u'paragraphs'][paragraph][u'qas'])
            for q in range(question_length):
                question_id = j[u'data'][data][u'paragraphs'][paragraph][u'qas'][q]["id"]

                question = corpus[counter]

                question_array = [question]

                question_pos = getPOS(question_array)

                # print (question_pos)
                q_word = ""
                q_verb = []
                q_noun = []
                for (w,t) in question_pos:
                    if w in question_words:
                        q_word = w
                    if t.startswith("V"):
                        q_verb.append(w)
                    if t.startswith("N"):
                        q_noun.append(w)
                
                # print ("Q-Word: " + q_word)
                # print ("Q-Verb: " + str(q_verb))
                # print ("Q-Noun: " + str(q_noun))


                # context_sentences = context.split(".")

                # temp_prediction = 0

                # for sentence in context_sentences:

                #     if (temp_prediction == 1):
                #         break

                #     words = nltk.word_tokenize(sentence.lower())

                #     if (bool(set(words) & set(q_verb)) == False):
                #         temp_prediction = 0
                #         break
                #     if (q_word in question_words):
                #         if (q_word == "who" or q_word == "Who"):
                #             #do NER
                #             chunks = nltk.chunk.util.tree2conlltags(nltk.ne_chunk(nltk.pos_tag(words)))
                #             if (len(chunks) > 0):
                #                 for chunk in chunks:
                #                     if (chunk[0] not in q_noun):
                #                         if (chunk[2] == "B-PERSON" or chunk[2] == "I-PERSON"):
                #                             temp_prediction = 1
                #                             break
                #                 break
                #             else:
                #                 temp_prediction = 1
                #         elif (q_word == "where" or q_word == "Where"):
                #             #do NER
                #             chunks = nltk.chunk.util.tree2conlltags(nltk.ne_chunk(nltk.pos_tag(words)))
                #             if (len(chunks) > 0):
                #                 for chunk in chunks:
                #                     if (chunk[0] not in q_noun):
                #                         if (chunk[2] == "B-GPE" or chunk[2] == "I-GPE"):
                #                             temp_prediction = 1
                #                             break
                #                 break
                #             else:
                #                 temp_prediction = 1
                #         elif (q_word == "when" or q_word == "When"):
                #             #do NER
                #             chunks = nltk.chunk.util.tree2conlltags(nltk.ne_chunk(nltk.pos_tag(words)))
                #             if (len(chunks) > 0):
                #                 for chunk in chunks:
                #                     if (chunk[0] not in q_noun):
                #                         if (chunk[2] == "B-DATE" or chunk[2] == "I-DATE"):
                #                             temp_prediction = 1
                #                             break
                #                 break
                #         else:
                #             #do POS
                #             for pos_tag in (nltk.pos_tag(words)):
                #                 if (pos_tag[1][0] == "N"):
                #                     if (pos_tag[0] not in q_noun):
                #                         temp_prediction = 1

                #     elif (q_word in question_words):
                #         temp_prediction = 1
                #         break
                
                # predictions[question_id] = temp_prediction

                # counter += 1


    print(predictions)

main()