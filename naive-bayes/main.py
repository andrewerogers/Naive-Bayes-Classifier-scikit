from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

NEGATIVE = 0
POSITIVE = 1
CELLS_LABELLED = open('sentiment labelled sentences/yelp_labelled.txt', 'r')

# build_review_label splits the file into features and reviews
def build_review_label(file):
    reviews = []
    labels = []
    for line in file:
        s = line.split('\t')
        if len(s) == 2:
            reviews.append(s[0])
            labels.append(s[1])
    return reviews, labels


# train_nb_model trains naive bayes model and displays accuracy
# metrics in cross balidation
def train_nb_model(reviews, review_labels):
    v = CountVectorizer(stop_words='english')
    all_features = v.fit_transform(reviews)
    x_train, x_test, y_train, y_test = train_test_split(all_features, review_labels,
                                                        test_size=0.1)
    nb = MultinomialNB()
    nb.fit(x_train, y_train)

    # accuracy
    num_corr = (y_test == nb.predict(x_test)).sum()
    num_inc = (y_test != nb.predict(x_test)).sum()
    acc = num_corr / (num_inc + num_corr)
    print(f"validation-accuracy: {acc * 100}%")
    return nb, v


# prompt_user for untrained reviews
def prompt_user(classifier, vectorizer):
    while True:
        example = input('> Enter a review: ')
        doc_term_matrix = vectorizer.transform([example])
        result = classifier.predict(doc_term_matrix)

        if int(result[0]) == POSITIVE:
            print('> Your review is positive.')
        else:
            print('> Your review is negative.')


reviews, review_labels = build_review_label(CELLS_LABELLED)
c, v = train_nb_model(reviews, review_labels)
prompt_user(c, v)


