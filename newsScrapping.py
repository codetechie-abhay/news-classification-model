import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import os

nltk.download('stopwords')
nltk.download('wordnet')

# Set the page title and icon for the Streamlit app
st.set_page_config(page_title="📰 News Category Prediction 📰", page_icon="📰")

# Function to load processed training data from a CSV file
def load_processed_data():
    df_processed = pd.read_csv("data/BBC_News_Train_Processed.csv")
    return df_processed

# Function to load testing data from a CSV file
def load_testing_data():
    df_test = pd.read_csv("data/BBC News Test.csv")
    return df_test

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase, remove stop words and lemmatize
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Function to train the model using the processed training data
def train_model():
    df = load_processed_data()
    df['Text'] = df['Text'].apply(preprocess_text)
    
    X_train, X_val, y_train, y_val = train_test_split(df['Text'], df['Category'], test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    # Hyperparameter tuning
    parameters = {
        'tfidfvectorizer__max_df': [0.75, 1.0],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'multinomialnb__alpha': [0.01, 0.1, 1.0]
    }
    grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # Evaluate the model
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return best_model, df, accuracy

# Function to map numeric labels to category names
def map_label_to_category(label):
    categories = {0: 'Business', 1: 'Entertainment', 2: 'Politics', 3: 'Sport', 4: 'Tech'}
    return categories[label]

# Function to scrape news headlines from a specified URL
def scrape_headlines():
    url = "https://www.indiatoday.in/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    headlines = []
    # Find all <h2> tags and extract their text content
    for item in soup.find_all('h2'):
        headlines.append(item.get_text())
    
    return headlines

# Function to save headlines to files based on their predicted categories
def save_headlines_to_files(headlines, predicted_categories):
    for headline, category in zip(headlines, predicted_categories):
        filename = f"{category.lower()}.txt"
        with open(filename, 'a') as file:
            file.write(headline + '\n')

# Main function to run the Streamlit app
def main():
    # App title and developer info
    st.title("📰 News Category Prediction App ")
    st.write("🛠️ Developed by Abhay Nautiyal 🛠️")

    # Display a news logo image
    st.image("news_logo.jpeg", use_column_width=True)

    # Load the testing data (not used in this example, but can be useful for further development)
    load_testing_data()

    # Train the model and get the trained model, data, and accuracy
    model, df, accuracy = train_model()

    # Display the model accuracy
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Display headlines section
    st.header("📰 Latest News Headlines 📰")
    headlines = scrape_headlines()
    st.write(headlines)
    
    # Button to predict categories for the scraped headlines
    if st.button("🔮 Predict Categories 🔮"):
        if headlines:
            predicted_categories = []
            # Predict category for each headline
            for headline in headlines:
                preprocessed_headline = preprocess_text(headline)
                prediction = model.predict([preprocessed_headline])[0]
                predicted_category = map_label_to_category(prediction)
                predicted_categories.append(predicted_category)
            
            # Save the headlines to files based on their predicted categories
            save_headlines_to_files(headlines, predicted_categories)
            
            # Display prediction results
            st.success("🎉 Predicted Categories: 🎉")
            for headline, category in zip(headlines, predicted_categories):
                st.write(f"{headline} -> {category}")
        else:
            st.warning("⚠️ No headlines found to predict. ⚠️")

    # Project details section
    st.markdown("---")
    st.header("📝 Project Details 📝")
    st.info(
        "This project predicts the category of news articles using a Multinomial Naive Bayes classifier.\n\n"
        "The model is trained on BBC news articles and their corresponding categories."
    )

if __name__ == "__main__":
    main()
