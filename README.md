## Aim
Predict the genre of a movie based on its plot summary and other features.

## Description
In this project, we utilize natural language processing (NLP) techniques for text classification on a movie dataset. The project involves cleaning and preprocessing text data, feature engineering, and building a machine learning model to predict movie genres.

## Technologies Used
Python
NLTK or SpaCy
Scikit-learn
What You Learn
Text classification with NLP
Text preprocessing
Feature engineering
Building and evaluating machine learning models
Dataset
The dataset contains the following features:

plot_summary: Summary of the movie plot
release_year: Year the movie was released
runtime: Duration of the movie
budget: Budget of the movie
box_office: Box office revenue
rating: Movie rating
num_reviews: Number of reviews
Project Implementation
1. Data Cleaning
The text data in the plot_summary column is cleaned by:

Removing non-alphanumeric characters
Removing extra spaces
Converting text to lowercase
2. Feature Engineering
The plot_summary column is transformed using TF-IDF vectorization. Numerical features such as release_year, runtime, budget, box_office, rating, and num_reviews are standardized.

3. Model Building
A Random Forest classifier is used to build the model. The data is split into training and testing sets, and the model is trained on the training data and evaluated on the testing data.

4. Evaluation
The model's performance is evaluated using accuracy and a classification report.

## Conclusion
This project provided hands-on experience with text classification using NLP techniques. By applying various preprocessing and feature engineering methods, we built a machine learning model capable of predicting movie genres based on plot summaries and other features.

Feel free to explore the repository and share your feedback!

And here is the link to the Google colab notebook:
https://colab.research.google.com/drive/10zheP_WycDdgcdFDKmb9320dNvKIFTZv?usp=sharing

![image](https://github.com/user-attachments/assets/72dc6f31-f966-429e-ad44-af0c092f1ada)
