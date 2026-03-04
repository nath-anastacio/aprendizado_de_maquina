#O exemplo a seguir prepara e transforma um conjunto de textos em um vetor numérico.

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

#Resultado:

['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

print(X.toarray())