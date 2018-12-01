import nltk
from collections import defaultdict
from collections import Counter

def countWordsInParagraph(context):
    wordList = nltk.word_tokenize(context)
    counts = Counter(wordList)
    return counts

def main():
    print (countWordsInParagraph("This is some \"paragraph."))

main()