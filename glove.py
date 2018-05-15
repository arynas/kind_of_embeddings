import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
# sentences and corpus from standard library
sentences = list(itertools.islice(Text8Corpus('text8'),None))
corpus = Corpus()
# fitting the corpus with sentences and creating Glove object
corpus.fit(sentences, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
# fitting to the corpus and adding standard dictionary to the object
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)