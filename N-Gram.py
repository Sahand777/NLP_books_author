  
score_table = np.zeros((20, 5))
score_table_train = np.zeros((20, 5))
for n in range(10, 200, 10): #number of words in each document
    df = pd.DataFrame(data={"token_words": ([None] * 1400), "Author":([None] * 1400)})
    df["token_words"] = df["token_words"].astype(object)
    for i in range(7):
        for j in range(200):
            rn = random.sample(range(0, len(df_books.at[i, "token_words"])), n)
            select_doc = []
            for r in rn:
                select_doc.append(df_books.at[i, "token_words"][r])
            df.at[((i*200)+j), "token_words"] = select_doc
            df.at[((i*200)+j), "Author"] = df_books.at[i, "Author"]
    
    # =============================================================================
    
    # =============================================================================
    # Removal of Punctuation Marks and Stopwords, and Lemmatisation of verbs
    # Using pre-defined function of text_filter
    df_joined = pd.DataFrame(index=range(1400), columns=["Text", "Author"])
    df_joined_filtered=pd.DataFrame(index=range(1400), columns=["Text", "Author"])
    for i in range(1400):
        df_joined.at[i,"Text"] = " ".join(df.at[i, "token_words"])
        df_joined.at[i,"Author"] = df.at[i, "Author"]
        df_joined_filtered.at[i,"Text"] = text_filter(df_joined.at[i,"Text"])
        df_joined_filtered.at[i,"Author"] = df.at[i, "Author"]
    # =============================================================================
    
    
    
    
    # =============================================================================
    # Label Encoding of Classes
    y = df_joined_filtered['Author']
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    # =============================================================================
    
    # =============================================================================
    # Splitting the dataset (80%: Training and 20%: Test)
    X = df_joined_filtered["Text"]
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                      ,test_size=0.2)
    cv = ShuffleSplit(n_splits = 10, test_size = 0.2)
    # =============================================================================
    
    # =============================================================================
    # N-Gram Transformation
    vectorizer = CountVectorizer(analyzer = "char_wb",
                                 ngram_range = (2,2))
    ngram_transformer = vectorizer.fit(X_train)
    text_ngram_train = ngram_transformer.transform(X_train)
    text_ngram_test = ngram_transformer.transform(X_test)
    
    # =============================================================================
    
    
    # =============================================================================
    # Classification Using Multinomial Naive Bayes
    classifier1 = MultinomialNB()
    classifier1 = classifier1.fit(text_ngram_train, y_train)
    score_table_train[int(n/10-1), 0] = classifier1.score(text_ngram_train, y_train)
    score1 = cross_val_score(classifier1, ngram_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 0] = stat.mean(score1)
    pred1 = classifier1.predict(text_ngram_test)
    pred_name1 = labelencoder.inverse_transform(pred1)
    cm1 = confusion_matrix(y_test, pred1)
    # =============================================================================
    
    # =============================================================================
    # Classification Using K Nearest Neighbors (K-NN)
    classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier2.fit(text_ngram_train, y_train)
    pred2 = classifier2.predict(text_ngram_test)
    pred_name2 = labelencoder.inverse_transform(pred2)
    score_table_train[int(n/10-1), 1] = classifier2.score(text_ngram_train, y_train)
    score2 = cross_val_score(classifier2, ngram_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 1] = stat.mean(score2)
    cm2 = confusion_matrix(y_test, pred2)
    # =============================================================================
    
    # =============================================================================
    # Classification Using Support Vector Machine (SVM)
    classifier3 = SVC(kernel = 'linear')
    classifier3.fit(text_ngram_train, y_train)
    pred3 = classifier3.predict(text_ngram_test)
    pred_name3 = labelencoder.inverse_transform(pred3)
    score_table_train[int(n/10-1), 2] = classifier3.score(text_ngram_train, y_train)
    score3 = cross_val_score(classifier3, ngram_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 2] = stat.mean(score3)
    cm3 = confusion_matrix(y_test, pred3)
    # =============================================================================
    
    # =============================================================================
    # Classification Using Decision Tree
    classifier4 = DecisionTreeClassifier() 
    classifier4.fit(text_ngram_train, y_train)
    pred4 = classifier4.predict(text_ngram_test)
    pred_name4 = labelencoder.inverse_transform(pred4)
    score_table_train[int(n/10-1), 3] = classifier4.score(text_ngram_train, y_train)
    score4 = cross_val_score(classifier4, ngram_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 3] = stat.mean(score4)
    cm4 = confusion_matrix(y_test, pred4)
    # =============================================================================
    
    # =============================================================================
    # Classification Using Random Forests
    classifier5 = RandomForestClassifier(n_estimators=100)
    classifier5.fit(text_ngram_train, y_train)
    pred5 = classifier5.predict(text_ngram_test)
    pred_name5 = labelencoder.inverse_transform(pred5)
    score_table_train[int(n/10-1), 4] = classifier5.score(text_ngram_train, y_train)
    score5 = cross_val_score(classifier5, ngram_transformer.transform(X), y, cv=cv)
    score_table[int(n/10-1), 4] = stat.mean(score5)
    cm5 = confusion_matrix(y_test, pred5)
    # =============================================================================
    
    # =============================================================================
    # Printing Results
    print("**Multinomial Naive Bayes***\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score1, n, classification_report(y_test, pred1)))
    print("***K Nearest Neighbors (K-NN)***\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score2, n, classification_report(y_test, pred2)))
    print("***Support Vector Machine (SVM)***\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score3, n, classification_report(y_test, pred3)))
    print("***Decision Tree***\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score4, n, classification_report(y_test, pred4)))
    print("***Random Forests***\n Score for n = {}: {}\n Classification Report for n = {}:\n {}.".
          format(n, score5, n, classification_report(y_test, pred5)))
    # =============================================================================
        
    from print_top10 import print_top10
    print('Top10 Words:\n')
    print_top10(ngram_transformer, classifier1, labelencoder.classes_)

# new_X = input("Enter the text from the book: ")
# new_X = pd.Series(new_X)
# new_X = pd.Series(text_filter(new_X))
# text_ngram_new_X = ngram_transformer.transform(new_X)
# new_pred = labelencoder.inverse_transform(classifier1.predict(text_ngram_new_X))
# print("The Author is {}".format(str(new_pred)))

# =============================================================================
# make a folder to save results
if not os.path.exists('./N-Gram_results'):
    os.makedirs('N-Gram_results')
# =============================================================================

# =============================================================================
# saving results
classifiers = ["Multinomial Naive Bayes", "K Nearest Neighbors", 
               "Support Vector Machine (SVM)", "Decision Tree",
               "Random Forests"] 
for i in range(0, 5):
    plt.plot()
    plt.plot(range(10, 200, 10), score_table[0:19, i], label="Test")
    plt.plot(range(10, 200, 10), score_table_train[0:19, i], label="Train")    
    plt.xlabel("Number of words in each document")
    plt.ylabel("Accuracy")
    plt.title("{} - N-Gram Error Analysis".format(classifiers[i]))
    plt.legend()
    plt.savefig("./N-Gram_results/{} - N-Gram Error Analysis.png".format(classifiers[i]))
    plt.show()
    plt.clf()
np.savetxt("./N-Gram_results/Score_N-gram_test.csv", score_table, delimiter=",")
np.savetxt("./N-Gram_results/Score_N-gram_Train.csv", score_table_train, delimiter=",")
# =============================================================================

