# 🧠 **DataBuddy**
### _Your Guide Through the World of Data..._

---

## 📌 **Project Overview**

**DataBuddy** is an interactive chatbot application designed to answer questions related to data science.  
It uses **Natural Language Processing (NLP)** techniques and a **Flask-based REST API** to process user queries and return accurate responses using a trained Machine Learning model.

---

## 🛠️ **Technologies & Tools Used**

- **🧠 Natural Language Processing (NLP)**:  
  Tokenization, cleaning, and preprocessing with `nltk`.

- **📊 Scikit-learn**:  
  `TfidfVectorizer` for converting text to numeric vectors and `MultinomialNB` for classification.

- **⚙️ Flask**:  
  Used to create a lightweight web API to process incoming questions.

- **🌐 HTML + JavaScript**:  
  Custom frontend with interactive UI and AJAX calls to Flask API.

- **📦 Joblib**:  
  To save and load the trained ML model.

---

## 💬 **NLP Concepts Implemented**

- Tokenization using `nltk.word_tokenize`
- Lowercasing and text cleaning
- TF-IDF vectorization
- Classification using Naive Bayes

---

## 🚀 **How Flask Was Helpful**

Flask provided:
- A clean and lightweight **REST API** (`/api/chatbot`) that connects the frontend and backend.
- A static file server to host the customized `index.html`.
- Simplicity and modularity for deploying and maintaining the app.

---

## 🖼️ **Screenshots**

### 🎯 User Interface

> A clean and modern interface for interacting with the chatbot.

![Chatbot API Response](images/sc_1.png)

---

## 📊 **Sample Dataset Preview**

> A snippet from the CSV file used to train the chatbot model (with Question-Answer pairs).

![Dataset Preview](images/data_1.png)

---

## 🔮 **Scope for Improvement**

Here’s how **DataBuddy** can be enhanced:

- 🧠 Use **BERT or GPT** for deeper language understanding.
- 💬 Add **contextual memory** for handling follow-up queries.
- 🌍 Support **multiple languages** using NLP translation models.
- 📱 Build a **mobile-friendly or app version** using Flutter or React Native.
- ⚙️ Create an **admin dashboard** to update Q&A without editing CSV files.

---

## 🏁 **Conclusion**

> **DataBuddy** brings together the power of **NLP, Machine Learning, Flask, and Web Development** into a clean, real-world chatbot application.  
It’s a stepping stone toward more intelligent, domain-specific assistants and has ample room for exciting extensions.

---

