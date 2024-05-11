def generate_report(cmatrix, score, creport):
  cmatrix = cmatrix.T
    plt.figure(figsize=(5,5))
  sns.heatmap(cmatrix, 
              annot=True, 
              fmt="d", 
              linewidths=.5, 
              square = True, 
              cmap = 'Blues', 
              annot_kws={"size": 16}, 
              xticklabels=['bad', 'good'],
              yticklabels=['bad', 'good'])

  plt.xticks(rotation='horizontal', fontsize=16)
  plt.yticks(rotation='horizontal', fontsize=16)
  plt.xlabel('Actual Label', size=20);
  plt.ylabel('Predicted Label', size=20);

  title = 'Accuracy Score: {0:.4f}'.format(score)
  plt.title(title, size = 20);

  print(creport)
  plt.show()



# Байесовский классификатор на основе матрицы признаков TF-IDF
mnb_tfidf = MultinomialNB(alpha=0.1)  
mnb_tfidf.fit(tfidf_X, labels)		

score_mnb_tfidf = mnb_tfidf.score(test_tfidf_X, test_labels)
predictions_mnb_tfidf = mnb_tfidf.predict(test_tfidf_X)
cmatrix_mnb_tfidf = confusion_matrix(test_labels, predictions_mnb_tfidf)
creport_mnb_tfidf = classification_report(test_labels, predictions_mnb_tfidf)

generate_report(cmatrix_mnb_tfidf, score_mnb_tfidf, creport_mnb_tfidf)

# Байесовский классификатор на основе признаков CountVectorizer
mnb_count = MultinomialNB()
mnb_count.fit(count_X, labels)

score_mnb_count = mnb_count.score(test_count_X, test_labels)
predictions_mnb_count = mnb_count.predict(test_count_X)
cmatrix_mnb_count = confusion_matrix(test_labels, predictions_mnb_count)
creport_mnb_count = classification_report(test_labels, predictions_mnb_count)

generate_report(cmatrix_mnb_count, score_mnb_count, creport_mnb_count)

# классификатор на основе алгоритма логистической регрессии, TF-IDF
lgs_tfidf = LogisticRegression(alpha = .1, solver='lbfgs')
lgs_tfidf.fit(tfidf_X, labels)

score_lgs_tfidf = lgs_tfidf.score(test_tfidf_X, test_labels)
predictions_lgs_tfidf = lgs_tfidf.predict(test_tfidf_X)
cmatrix_lgs_tfidf = confusion_matrix(test_labels, predictions_lgs_tfidf)
creport_lgs_tfidf = classification_report(test_labels, predictions_lgs_tfidf)

generate_report(cmatrix_lgs_tfidf, score_lgs_tfidf, creport_lgs_tfidf)

# Классификатор логистической регрессии на основе признаков #CountVectorizer
lgs_count = LogisticRegression(solver='lbfgs')
lgs_count.fit(count_X, labels)

score_lgs_count = lgs_count.score(test_count_X, test_labels)
predictions_lgs_count = lgs_count.predict(test_count_X)
cmatrix_lgs_count = confusion_matrix(test_labels, predictions_lgs_count)
creport_lgs_count = classification_report(test_labels, predictions_lgs_count)
generate_report(cmatrix_lgs_count, score_lgs_count, creport_lgs_count)

