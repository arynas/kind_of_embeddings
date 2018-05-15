from sklearn.feature_extraction.text import TfidfTransformer
# create tf-idf object
transformer = TfidfTransformer(smooth_idf=False)
# X can be obtained as X.toarray() from the previous snippet
X = [[3, 0, 1],
     [5, 0, 0],
     [3, 0, 0],
     [1, 0, 0],
     [3, 2, 0],
     [3, 0, 4]]
# learn the vocabulary and store tf-idf sparse matrix in tfidf
tfidf = transformer.fit_transform(counts)
# retrieving matrix in numpy form as we did it before
tfidf.toarray()