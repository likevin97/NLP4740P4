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

def getQuestionWord(question, question_words):
    question_array = [question]
    question_pos = getPOS(question_array)
    q_word = "<UNK>"

    for (w,t) in question_pos:
        if w.lower() in question_words:
            q_word = w
            break
    
    return q_word

def getQuestionVerbs(question):
    question_array = [question]
    question_pos = getPOS(question_array)
    q_verbs = []

    for (w,t) in question_pos:
        if t.startswith("V"):
            q_verbs.append(w)
    
    return q_verbs

def getQuestionNouns(question):
    question_array = [question]
    question_pos = getPOS(question_array)
    q_nouns = []
    
    for (w,t) in question_pos:
        if t.startswith("N"):
            q_nouns.append(w)
    
    return q_nouns

def getVerbCount(question, context):

    q_verbs = getQuestionVerbs(question.lower())

    pos_dict = {}

    pos = getPOS([context.lower(), question.lower()])

    for (word, tag) in pos:
        wtag = tag[0].lower()
        wtag = wtag if wtag in ["a","r","n","v"] else None
        pos_dict[word] = wtag
    
    lemmatized = ourLemmatize(pos_dict, [context.lower(), question.lower()])

    return len(set(nltk.word_tokenize(lemmatized[0])) & set(q_verbs))

def getNounCount(question, context):
    q_nouns = getQuestionNouns(question.lower())

    pos_dict = {}

    pos = getPOS([context.lower(), question.lower()])

    for (word, tag) in pos:
        wtag = tag[0].lower()
        wtag = wtag if wtag in ["a","r","n","v"] else None
        pos_dict[word] = wtag
    
    lemmatized = ourLemmatize(pos_dict, [context.lower(), question.lower()])

    return len(set(nltk.word_tokenize(lemmatized[0])) & set(q_nouns))




def getNERCount(question_words, question, context, qwordTagMap):
    '''
    for the question word, does the corresponding NER tag exist in the context.
    For every NER tag exists, see if that word is in the question
    if yes dont count, else count
    '''

    q_word = getQuestionWord(question, question_words)

    print(q_word)
    print(question)

    qwordTags = qwordTagMap[q_word.lower()]

    num_of_tags_in_context = 0


    question_words = nltk.word_tokenize(question)

    question_chunks = nltk.chunk.util.tree2conlltags(nltk.ne_chunk(nltk.pos_tag(question_words)))

    questionNERWords = []

    if (len(question_chunks) > 0):
        for question_chunk in question_chunks:
            questionNERWords.append(question_chunk[0])


    context_words = nltk.word_tokenize(context)

    context_chunks = nltk.chunk.util.tree2conlltags(nltk.ne_chunk(nltk.pos_tag(context_words)))

    if (len(context_chunks) > 0):
        for context_chunk in context_chunks:
            if (context_chunk[2] in qwordTags and context_chunk[0] not in questionNERWords):
                num_of_tags_in_context += 1

    return num_of_tags_in_context




def main():
    stop_words = set(stopwords.words('english'))
    # stop_words.union(set(string.punctuation))
    corpus = createCorpus(stop_words, "training_sample.json")
    #print (corpus)
    #print ("=---=-------")

    file = open("training_sample.json")
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
    whoTags = ["B-PERSON", "I-PERSON"]
    whatTags = ["B-GPE", "I-GPE", "B-LOCATION", "I_LOCATION"]
    whenTags = ["B-DATE", "I-DATE", "B-TIME", "I-TIME"]
    otherTags = ["B-PERSON", "I-PERSON", "B-GPE", "I-GPE", "B-LOCATION", "I-LOCATION", "B-DATE", "I-DATE", "B-TIME", "I-TIME"]

    qwordTagMap = {}

    for question_word in question_words:
        qwordTagMap[question_word] = otherTags
    qwordTagMap["who"] = whoTags
    qwordTagMap["what"] = whatTags
    qwordTagMap["when"] = whenTags
    qwordTagMap["<unk>"] = otherTags

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
            

                # nerCounts = getNERCount(question_words, question, context, qwordTagMap)

                verbCount = getVerbCount(question, context)
                nounCount = getNounCount(question, context)

                # print(nerCounts)
                print(verbCount)
                print(nounCount)

                counter += 1


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


    # print(predictions)

main()