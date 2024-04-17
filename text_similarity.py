# text_similarity.py

# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Function to compute cosine similarity between two text paragraphs using TF-IDF
def compute_cosine_similarity(text1, text2):
    """
    Compute the cosine similarity between two text strings.

    Args:
    text1 (str): The first text string.
    text2 (str): The second text string.

    Returns:
    float: The cosine similarity score between the two text strings.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0][0]

# Load the dataset
data = pd.read_csv('DataNeuron_Text_Similarity.csv')  # Adjust path as needed

# Split the data into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Apply the cosine similarity function to each pair of text paragraphs in the test dataset
test_data['STS'] = test_data.apply(lambda row: compute_cosine_similarity(row['text1'], row['text2']), axis=1)

# Save the results to a CSV file
test_data[['text1', 'text2', 'STS']].to_csv('output.csv', index=False)

print("Output saved to output.csv")

def train_and_save_model(data_path):
    data = pd.read_csv(data_path)
    # Assuming 'text1' and 'text2' are the columns with texts
    text_data = data['text1'].tolist() + data['text2'].tolist()  # Combine text for training
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text_data)

    # Save the trained vectorizer to a pickle file
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Model saved as 'tfidf_vectorizer.pkl'")

if __name__ == "__main__":
    train_and_save_model('DataNeuron_Text_Similarity.csv')
