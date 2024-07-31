import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to clean text
def clean_text_simple(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

# Load dataset
df = pd.read_csv('movies_dataset.csv')

# Apply text cleaning to the plot_summary column
df['clean_plot_summary'] = df['plot_summary'].apply(clean_text_simple)

# Encode the target variable (genre)
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['genre'])

# Define features and target variable
features = df[['clean_plot_summary', 'release_year', 'runtime', 'budget', 'box_office', 'rating', 'num_reviews']]
target = df['genre_encoded']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Define a column transformer to apply TF-IDF to the plot summary and scale other features
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', tfidf_vectorizer, 'clean_plot_summary'),
        ('num', StandardScaler(), ['release_year', 'runtime', 'budget', 'box_office', 'rating', 'num_reviews'])
    ]
)

# Preprocess the training and testing data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Initialize Random Forest Classifier
random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
random_forest_clf.fit(X_train_transformed, y_train)

# Predict on the test set
y_pred = random_forest_clf.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
