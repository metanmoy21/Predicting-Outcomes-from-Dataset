# Predicting-Outcomes-from-Dataset

**Company** - CODETECH IT SOLUTIONS

**Name** - Tanmoy Das

**Intern ID** - CT06DR1738

**Domain** - Python Programming

**Duration** - 6 weeks

**Mentor** - Neela Santosh

This Python script implements a Spam Detection model using the Scikit-learn library, which is one of the most popular tools for machine learning in Python. The primary goal of this project is to classify text messages as either “spam” or “ham” (non-spam). Spam detection is a real-world application of Natural Language Processing (NLP) and machine learning, and it is widely used in email systems, messaging apps, and online platforms to filter out unwanted or potentially harmful messages.

**Tools and Libraries Used**

* **Pandas** - Pandas is a powerful library for data manipulation and analysis. In this script, Pandas is used to create a DataFrame from a dictionary of sample messages and their corresponding labels (“spam” or “ham”). It allows structured handling of the dataset and makes it easy to separate features (the message text) from labels (the classification target).

* **Scikit-learn** - Scikit-learn is a robust machine learning library in Python that provides easy-to-use tools for building predictive models. Several modules from Scikit-learn are used in this script:

* **train_test_split** - This function splits the dataset into training and testing sets, ensuring that the model can be evaluated on unseen data.

* **CountVectorizer** - This module converts text data into a numerical format that machine learning algorithms can understand. It transforms each message into a bag-of-words vector, counting the frequency of each word in the dataset.

* **MultinomialNB** - This is the Naive Bayes classifier specifically designed for discrete data like word counts. It is widely used for text classification tasks such as spam detection because it performs well with relatively small datasets and high-dimensional feature spaces.

* **accuracy_score** - This metric evaluates the model’s performance by comparing predicted labels with the true labels of the test data.

**How the Script Works**

* **Dataset Creation** - The script starts with a small dataset of 20 messages, each labeled as either “spam” or “ham.” Spam messages include promotional content, lottery notifications, and offers, while ham messages are normal conversations or reminders. This dataset is loaded into a Pandas DataFrame, with X representing the messages and y representing the labels.

* **Text Vectorization** - Machine learning models cannot work with raw text, so it must be converted into numerical features. The CountVectorizer is used here, which tokenizes the text (splits it into words) and converts each message into a vector of word counts. Each column in the resulting matrix represents a unique word, and each row represents a message.

* **Train-Test Split** -The dataset is divided into training and testing sets with a 70-30 split. This ensures that the model is trained on most of the data but is evaluated on unseen data to assess its real-world performance.

* **Training the Naive Bayes Classifier** - The Multinomial Naive Bayes model is trained on the vectorized training data. Naive Bayes is particularly suitable for spam detection because it assumes independence between features (words) and is very efficient for high-dimensional data like text.

* **Model Evaluation** - After training, the model predicts labels for the test set. The accuracy_score function calculates the percentage of correctly classified messages. This gives an initial measure of how well the model can distinguish spam from ham.

* **Making Predictions on New Messages** - The script also demonstrates how to use the trained model to classify new messages. Messages such as “You have won a free gift” and “Don’t forget our meeting tomorrow” are vectorized using the same CountVectorizer and then classified. The model predicts whether each message is spam or ham.

**Real-Life Applications**

Spam detection is a crucial application of machine learning in everyday life -

* **Email Filtering** - Services like Gmail and Outlook rely on similar models to automatically detect and filter spam messages.

* **Messaging Apps** - Platforms like WhatsApp, Messenger, and Telegram use machine learning to warn users about suspicious messages.

* **Cybersecurity** - Spam detection models help prevent phishing attempts, malware delivery, and fraudulent messages.

* **Marketing** - Companies can analyze messages to understand which types of messages are considered spam by users.

**Conclusion**

In summary, this script provides a simple but effective implementation of a text classification system for spam detection. By combining Pandas for data handling, CountVectorizer for text vectorization, and Multinomial Naive Bayes for classification, it demonstrates a typical workflow for solving real-world NLP problems. While the dataset is small, the approach can easily be scaled to handle thousands or millions of messages, making it highly relevant in practical applications such as email clients, messaging platforms, and cybersecurity solutions.
