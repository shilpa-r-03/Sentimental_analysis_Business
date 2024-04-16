import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer

# Define the preprocessing function
def preprocessor(reviews):
    # Define HTMLTAGS for removing HTML tags
    HTMLTAGS = re.compile('<.*?>')
    
    # Define table for removing punctuation
    table = str.maketrans(dict.fromkeys(string.punctuation))
    
    # Define remove_digits for removing digits
    remove_digits = str.maketrans('', '', string.digits)
    
    # Define MULTIPLE_WHITESPACE for replacing multiple whitespaces with single space
    MULTIPLE_WHITESPACE = re.compile(r"\s+")
    
    # Define stopwords
    total_stopwords = set(stopwords.words('english'))
    negative_stop_words = set(word for word in total_stopwords if "n't" in word or 'no' in word)
    final_stopwords = total_stopwords - negative_stop_words
    final_stopwords.add("one")
    
    # Create stemming object
    stemmer = PorterStemmer()

    # Remove HTML tags
    reviews = HTMLTAGS.sub(r'', reviews)

    # Remove punctuation
    reviews = reviews.translate(table)
    
    # Remove digits
    reviews = reviews.translate(remove_digits)
    
    # Lowercase all letters
    reviews = reviews.lower()
    
    # Replace multiple white spaces with single space
    reviews = MULTIPLE_WHITESPACE.sub(" ", reviews).strip()
    
    # Remove stop words
    reviews = [word for word in reviews.split() if word not in final_stopwords]
    
    # Stemming
    reviews = ' '.join([stemmer.stem(word) for word in reviews])
    
    return reviews

# Load the saved model and vectorizer
with open("transformer1.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)

with open("model1.pkl", "rb") as f:
    loaded_model = pickle.load(f)

def predict_sentiment(review):
    # Preprocess the review
    preprocessed_review = preprocessor(review)
    # Vectorize the preprocessed review
    vectorized_review = loaded_vectorizer.transform([preprocessed_review])
    # Predict sentiment
    sentiment_label = loaded_model.predict(vectorized_review)[0]
    
    # Map numerical label to text label
    if sentiment_label == 0:
        return "Negative"
    elif sentiment_label == 1:
        return "Neutral"
    else:
        return "Positive"

new_review = "Average experience overall. The product worked fine, but nothing remarkable."
predicted_sentiment = predict_sentiment(new_review)
print(f"The predicted sentiment for the review is: {predicted_sentiment}")