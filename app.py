import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Page Title and Icon
st.set_page_config(page_title="📰 News Category Prediction 📰", page_icon="📰")

# Function to load processed training data
def load_processed_data():
    df_processed = pd.read_csv("data/BBC_News_Train_Processed.csv")
    return df_processed

# Function to load testing data
def load_testing_data():
    df_test = pd.read_csv("data/BBC News Test.csv")
    return df_test

# Function to train the model
def train_model():
    df = load_processed_data()
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['Text'], df['Category'])
    return model, df

# Function to map numeric labels to category names
def map_label_to_category(label):
    categories = {0: 'Business', 1: 'Entertainment', 2: 'Politics', 3: 'Sport', 4: 'Tech'}
    return categories[label]

# Main function to run the app
def main():
    # App title and developer info
    st.title("📰 News Category Prediction App ")
    st.write("🛠️ Developed by Abhay Nautiyal 🛠️")

    # News Logo
    st.image("news_logo.jpeg", use_column_width=True)

    # Load the testing data
    load_testing_data()

    # Train the model
    model, df = train_model()

    # Text input for user to enter text
    st.header("✍️ Enter Text ✍️")
    text = st.text_area("", "Write a news article here...")

    # Predict button
    if st.button("🔮 Predict 🔮"):
        if text:
            # Make prediction
            prediction = model.predict([text])[0]
            predicted_category = map_label_to_category(prediction)

            # Display prediction result
            st.success(f"🎉 The predicted category is: {predicted_category} 🎉")
        else:
            st.warning("⚠️ Please enter some text before predicting. ⚠️")

    # Project details
    st.markdown("---")
    st.header("📝 Project Details 📝")
    st.info(
        "This project predicts the category of news articles using a Multinomial Naive Bayes classifier.\n\n"
        "The model is trained on BBC news articles and their corresponding categories."
    )

if __name__ == "__main__":
    main()

