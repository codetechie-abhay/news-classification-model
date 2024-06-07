# 📰 News Category Prediction App

This Streamlit app predicts the category of news articles based on their content. It uses a machine learning algorithm called Multinomial Naive Bayes, which is a common algorithm for text classification tasks.

## ℹ️ How it Works

1. **Input Text**: Users can enter text into the provided text area.
2. **Prediction**: After entering the text, users can click the "Predict" button.
3. **Output**: The app then predicts the category of the input text and displays the result.

## 📝 Example

Let's say you have a piece of text:

```
1. "Apple is expected to release a new iPhone model next month."

2. "Janvi Kapoor On Entering South Indian Cinema
In a news report by PTI Janhvi stated, "Somehow it makes me feel closer to my mom, to be in that environment, as well as to hear and speak in that language. It just felt like it was the right time, I felt I was gravitating towards it (sic)." She further added, "Mom has such a history with the families of NTR sir and Ram Charan sir, it's my honour that I'm able to work with both these extremely talented actors," the actor, whose latest release is "Mr. & Mrs. Mahi (sic)."
"

3. "FM Nirmala Sitharaman presents Budget 2024"

4. "AI in iOS 18 could be a privacy concern for some people but Apple has a plan"
```

You input this text into the app, click "Predict", and the app predicts the category of the text, which might be "Technology" in this case.

## 🧠 Algorithm

The algorithm used in this app is Multinomial Naive Bayes:

1. **Data Preparation**: The training data consists of news articles with their corresponding categories.
2. **Feature Extraction**: Text data is converted into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
3. **Model Training**: A Multinomial Naive Bayes classifier is trained on the TF-IDF vectors and their corresponding categories.
4. **Prediction**: When a new text is input, the trained model predicts its category based on the learned patterns in the training data.

## 🚀 Running the App

To run the app:

1. Install the required Python libraries:

    ```
    pip install streamlit pandas scikit-learn
    ```

2. Run the Streamlit app:

    ```
    streamlit run app.py
    ```

3. Enter text into the text area, click "Predict", and see the predicted category.

---

Developed by Abhay Nautiyal