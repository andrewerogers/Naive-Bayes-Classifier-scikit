from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

NEGATIVE = 0
POSITIVE = 1
AMAZON_CELLS_LABELLED = open('sentiment labelled sentences/amazon_cells_labelled.txt', 'r')

amazon_reviews = []
review_labels = []
for feature in AMAZON_CELLS_LABELLED:
    s = feature.split('\t')
    if len(s) == 2:
        amazon_reviews.append(s[0])
        review_labels.append(s[1])

# build vocabulary / data sets
v = CountVectorizer(stop_words='english')
all_features = v.fit_transform(amazon_reviews)
X_train, X_test, y_train, y_test = train_test_split(all_features, review_labels,
                                                    test_size=0.1, random_state=75)
nb = MultinomialNB()
nb.fit(X_train, y_train)

# model validation and testing - normally we would incorporate more here
num_corr = (y_test == nb.predict(X_test)).sum()
num_inc = (y_test != nb.predict(X_test)).sum()
acc = num_corr / (num_inc + num_corr)

# prompt user for untrained reviews
while True:
    example = input('> Enter a review: ')
    doc_term_matrix = v.transform([example])
    result = nb.predict(doc_term_matrix)

    if int(result[0]) == POSITIVE:
        print('> Your review is positive.')
    else:
        print('> Your review is negative.')
