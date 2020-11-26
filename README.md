# NLP_books_author
This piece of code is designed to download seven books from Gutenberg online library and identify the author of the books after fitting a classification model.
There are six different python files. For excuting the code, running the main.py is enough. 
## 1	Introduction
In this project, the goal is to identify the author of several books by analyzing some words from the original books. Three different Feature selection methods for texts and five different classifying methods are being used.
The ultimate target is to classify, predict, and compare the methods. 
## 2	Data Explanation and Preparation 
Seven different books are selected from Gutenberg Digital Books Library. The books are listed below:
* Three Musketeers by Alexandre Dumas, Pere
* Adventures of Sherlock Holmes, by A. Conan Doyle
* Dorothy and the Wizard in Oz, by L. Frank Baum.
* The Mysterious Island, by Jules Verne
* Mechanical Drawing Self-Taught, by Joshua Rose
* The History and Practice of the Art of Photography, by Henry H. Snelling
* Christmas Carol, by Charles Dicken
Firstly, since each text is downloaded from Project Gutenberg contains a header with the name of the author, the names of people who scanned and corrected it, a license, and so on, the beginning and the end is removed, to be just the content and nothing else.
Then, the books is tokenized to preprocess the texts. the text is filtered by removing punctuations and stopwords and Lemmatization by using the NLTK library.
Then, a DataFrame contains 200 records is made; each includes n words from each book. There is a loop for n from 10 to 200 with steps of 10. Making this loop help to show that how the model get better when there are more words for each records.
## 3	Feature Engineering
Text and text documents are unstructured and therefore need to be transformed into computer-structured and computable data by various operations (categorized as pre-processing). Three different methods of feature engineering of texts are used. 
* Bag of Words
* TF-IDF
* N-Gram
## 4	Modeling
Five classification methods have been used in our project.
* Multinomial Naive Bayes
* K-Nearest Neighbors (K-NN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
For comparing the models and choosing the champion model, cross-validation scores, and the mean of these ten scores is used as the final indicator.



