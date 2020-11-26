from urllib import request
import nltk 
import nltk.corpus  
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import random
from text_filter import text_filter
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import statistics as stat
import os
import sys
import nltk 
import nltk.corpus  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

abspath = os.path.abspath(sys.argv[0])
dname = os.path.dirname(abspath)
os.chdir(dname)

method = input("Select the feature Engineering method: (1)Bag of Words, (2)TF-IDF, (3)N-Gram, (4)All\n")


# =============================================================================
# Getting the 1st book from the Gutenberg library 
#The Three Musketeers by Alexandre Dumas, Pere
url = "https://www.gutenberg.org/files/1257/1257-0.txt"
response = request.urlopen(url)
TheThreeMusketeers = response.read().decode('utf8')
TheThreeMusketeers = TheThreeMusketeers[
        TheThreeMusketeers.find("AUTHOR’S PREFACE"):
            TheThreeMusketeers.rfind("END OF THIS PROJECT GUTENBERG EBOOK")]
tokens_TheThreeMusketeers = word_tokenize(TheThreeMusketeers)
text_TheThreeMusketeers = nltk.Text(tokens_TheThreeMusketeers)

# =============================================================================

# =============================================================================
# Getting the 2nd book from the Gutenberg library 
#Adventures of Sherlock Holmes, by A. Conan Doyle
url = "https://www.gutenberg.org/files/1661/1661-0.txt"
response = request.urlopen(url)
SherlockHolmes = response.read().decode('utf8')


SherlockHolmes = SherlockHolmes[SherlockHolmes.find("I. A SCANDAL IN BOHEMIA"):
    SherlockHolmes.rfind("End of Project Gutenberg's")]
tokens_SherlockHolmes = word_tokenize(SherlockHolmes)
text_SherlockHolmes = nltk.Text(tokens_SherlockHolmes)

# =============================================================================

# =============================================================================
#  Getting the 3rd book from the Gutenberg library 
#Dorothy and the Wizard in Oz, by L. Frank Baum.
url = "http://www.gutenberg.org/cache/epub/420/pg420.txt"
response = request.urlopen(url)
Oz = response.read().decode('utf8')


Oz = Oz[Oz.find("To My Readers"):
    Oz.rfind("End of Project Gutenberg's")]
tokens_Oz = word_tokenize(Oz)
text_Oz = nltk.Text(tokens_Oz)
# =============================================================================

# =============================================================================
#  Getting the 4th book from the Gutenberg library 
# The Mysterious Island, by Jules Verne
url = "https://www.gutenberg.org/files/1268/1268-0.txt"
response = request.urlopen(url)
MysteriousIsland = response.read().decode('utf8')


MysteriousIsland = MysteriousIsland[
        MysteriousIsland.find("PART 1--DROPPED FROM THE CLOUDS"):
            MysteriousIsland.rfind("End of the Project Gutenberg EBook")]
tokens_MysteriousIsland = word_tokenize(MysteriousIsland)
text_MysteriousIsland = nltk.Text(tokens_MysteriousIsland)
# =============================================================================

# =============================================================================
#  Getting the 5th book from the Gutenberg library
# Mechanical Drawing Self-Taught, by Joshua Rose
url = "http://www.gutenberg.org/cache/epub/23319/pg23319.txt"
response = request.urlopen(url)
MechanicalDrawing = response.read().decode('utf8')


MechanicalDrawing = MechanicalDrawing[
        MechanicalDrawing.find("_THE DRAWING BOARD._"):
            MechanicalDrawing.rfind("End of Project Gutenberg's")]
tokens_MechanicalDrawing = word_tokenize(MechanicalDrawing)
text_MechanicalDrawing = nltk.Text(tokens_MechanicalDrawing)
# =============================================================================

# =============================================================================
# Getting the 6th book from the Gutenberg library
# The History and Practice of the Art of Photography, by Henry H. Snelling
url = "http://www.gutenberg.org/cache/epub/168/pg168.txt"
response = request.urlopen(url)
ArtofPhotography = response.read().decode('utf8')


ArtofPhotography = ArtofPhotography[
        ArtofPhotography.find("INTRODUCTION"):
            ArtofPhotography.rfind("End of the Project Gutenberg EBook")]
tokens_ArtofPhotography = word_tokenize(ArtofPhotography)
text_ArtofPhotography = nltk.Text(tokens_ArtofPhotography)
# =============================================================================

# =============================================================================
#  Getting the 7th book from the Gutenberg library
# ChristmasCarol, by Charles Dickens
url = "https://www.gutenberg.org/files/24022/24022-0.txt"
response = request.urlopen(url)
ChristmasCarol = response.read().decode('utf8')


ChristmasCarol = ChristmasCarol[
        ChristmasCarol.find("Marley was dead, to begin with"):
            ChristmasCarol.rfind("End of the Project Gutenberg EBook")]
tokens_ChristmasCarol = word_tokenize(ChristmasCarol)
text_ChristmasCarol = nltk.Text(tokens_ChristmasCarol)

# =============================================================================

# =============================================================================
books= [tokens_ArtofPhotography,
        tokens_MechanicalDrawing,
        tokens_MysteriousIsland,
        tokens_Oz,
        tokens_ChristmasCarol,
        tokens_SherlockHolmes,
        tokens_TheThreeMusketeers]

# =============================================================================
    


# =============================================================================
# Making the dataset
df_books = pd.DataFrame(data={"token_words": [books[0], 
                                              books[1],
                                              books[2],
                                              books[3],
                                              books[4],
                                              books[5],
                                              books[6]],
                                        "Author": ["Henry H. Snelling",
                                                       "Joshua Rose",
                                                       "Jules Verne",
                                                       "L. Frank Baum",
                                                       "Charles Dickens",
                                                       "A. Conan Doyle",
                                                       "Alexandre Dumas"]})
exec(open("./WordCloud.py").read())
n_start = 10
n_end = 201
n_step = 10    


if method == 1:
    exec(open("./BoW.py").read())
elif method == 2:
    exec(open("./Tfidf.py").read())
elif method == 3:
    exec(open("./N-Gram.py").read())
else:
    print('******Bags of Words******')
    exec(open("./BoW.py").read())
    print('******End of Bags of Words******')
    print('******TFIDF******')
    exec(open("./Tfidf.py").read())
    print('******End of TFIDF******')
    print('******N-Gram******')
    exec(open("./N-Gram.py").read())
    print('******N-Gram******')

