# **Random Forest for Amazon Reviews Classification**  

**Random Forest for Amazon Reviews Classification** is a machine learning project that utilizes **Random Forest classifiers** to categorize Amazon product reviews as **positive or negative** based on textual content. The model is trained using **supervised learning techniques** and evaluated for accuracy in sentiment analysis.

---

## **Table of Contents**  

1. [Project Overview](#project-overview)  
2. [Video Demo](#video-demo)  
3. [Motivation and Purpose](#motivation-and-purpose)  
4. [Problem Statement and Objectives](#problem-statement-and-objectives)  
5. [Dataset Description](#dataset-description)  
6. [Model Architecture](#model-architecture)  
7. [Installation and Usage](#installation-and-usage)  
8. [Evaluation and Results](#evaluation-and-results)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## **Project Overview**  

### **Introduction**  
This project explores how **Random Forest** can be used for **sentiment analysis** on Amazon product reviews. The goal is to build a classification model that accurately predicts whether a review is **positive or negative** based on its text.

---

## **Video Demo**  

🔗 *[Include link to video demo if available]*  

---

## **Motivation and Purpose**  
Analyzing customer sentiment is essential for e-commerce platforms to understand user feedback and improve product offerings. This project aims to:  

- Gain hands-on experience with **Random Forest** for text classification.  
- Explore **Natural Language Processing (NLP)** techniques for sentiment analysis.  
- Evaluate the effectiveness of **Random Forest classifiers** in predicting sentiment.  

---

## **Problem Statement and Objectives**  

### **Problem Statement:**  
Customer reviews contain valuable insights, but manually analyzing thousands of reviews is inefficient. The objective of this project is to build an **automated sentiment classification model** that accurately predicts whether a review is **positive** or **negative** based on textual content.

### **Objectives:**  
✔️ Preprocess and clean **Amazon review data** for analysis.  
✔️ Train a **Random Forest classifier** to categorize reviews.  
✔️ Experiment with different **feature extraction techniques** (e.g., TF-IDF, Count Vectorization).  
✔️ Evaluate model performance using **classification metrics** (accuracy, precision, recall, F1-score).  

---

## **Dataset Description**  
The dataset consists of Amazon product reviews, including:

| Feature | Description |  
|---------|------------|  
| **Rating** | Star rating given by the customer |  
| **Date** | Date the review was submitted |  
| **Variation** | Product variation (e.g., size, color) |  
| **Verified Reviews** | The actual review text provided by the customer |  
| **Feedback** | Customer feedback on the helpfulness of the review |  
| **Sentiment Label** | Binary classification (0 = Negative, 1 = Positive) |  

📌 **Data Preprocessing Steps:**  
- **Text Cleaning:** Removal of stopwords, punctuation, and special characters.  
- **Tokenization:** Splitting sentences into words.  
- **Vectorization:** Converting text into numerical form using **TF-IDF** or **Count Vectorization**.  

---

## **Model Architecture**  
The model uses a **Random Forest Classifier** trained on preprocessed text features. The workflow includes:

1. **Text Preprocessing** – Cleaning and transforming review text.  
2. **Feature Extraction** – Converting text into numerical vectors.  
3. **Training the Random Forest Model** – Learning patterns in the data.  
4. **Evaluating Model Performance** – Assessing accuracy and classification metrics.

📌 **Hyperparameters Used:**  
- **Number of Trees**: Tuned for best performance  
- **Criterion**: Gini Impurity / Entropy  
- **Max Depth**: Adjusted to prevent overfitting  
- **Min Samples Split**: Optimized for efficiency  

---

## **Installation and Usage**  

### **🔧 Prerequisites**  
Install the necessary dependencies before running the project:  

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

### **📌 Running the Model**  

#### **Clone the Repository:**  
```bash
git clone https://github.com/MilkyBenji/ML_Project_2.git
cd ML_Project_2
```

#### **Run the Jupyter Notebook:**  
```bash
jupyter notebook Random_Forest_Amazon_Reviews.ipynb
```

#### **Train the Model and Evaluate Predictions**  

---

## **Evaluation and Results**  

### **📊 Model Performance**  
The **Random Forest model** was trained and evaluated using various classification metrics:

- **Accuracy:** Measures overall correctness.  
- **Precision & Recall:** Determines model effectiveness in classifying reviews correctly.  
- **Confusion Matrix:** Provides a visual breakdown of predictions.

📌 **Visualization:**  
- A **classification report** shows how well the model predicts positive and negative reviews.  
- The **decision tree diagram** provides insights into how the model makes decisions.  

### **📈 Sample Prediction**  

Given a sample review:
```python
sample_review = ["This product is amazing! Works perfectly."]
```

The model predicts:
```bash
Predicted Sentiment: Positive
```

---

## **Contributing**  

If you'd like to contribute to **Random Forest for Amazon Reviews Classification**, feel free to **fork the repository** and submit a **pull request**. Contributions are always welcome!  

### **Guidelines:**  
✔️ **Follow Best Practices**: Ensure the code is clean and well-documented.  
✔️ **Testing**: Validate model performance before submitting any changes.  
✔️ **Feature Additions**: If suggesting enhancements, provide a detailed explanation.  

---

## **License**  

This project is licensed under the **MIT License** – see the `LICENSE` file for details.  

---

## **📌 Summary**  
🚀 This project applies **Random Forest** to classify Amazon reviews as **positive or negative** using **Natural Language Processing (NLP)** techniques. By leveraging **machine learning**, it provides an **automated approach** to analyzing customer sentiment efficiently.

