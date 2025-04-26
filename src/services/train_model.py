import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from tqdm import tqdm
import joblib
import os

print("IS RUNNING")

# Define paths
data_dir = "data"  # Adjust if you move it elsewhere
ratings_path = os.path.join(data_dir, "ratings.csv")

# Load and prepare ratings data
ratings = pd.read_csv(ratings_path)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split and model training
trainset, testset = train_test_split(data, test_size=0.2)
print("Training model...")
model = SVD()
for _ in tqdm(range(1), desc="Fitting model"):  # One iteration, tqdm used just to show progress
    model.fit(trainset)

# Evaluate the model
print("\nEvaluating model on test set:")
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# Save trained model
os.makedirs("backend/src/services", exist_ok=True)
joblib.dump(model, "backend/src/services/recommender_model.pkl")

print("Model trained and saved successfully.")
