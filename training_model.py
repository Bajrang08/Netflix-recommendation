import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Data
df = pd.read_csv('netflix_titles (1).csv')

# 2. Preprocessing
# Filter for Movies only and extract numerical duration
movies_df = df[df['type'] == 'Movie'].copy()
movies_df['duration_min'] = movies_df['duration'].str.extract('(\d+)').astype(float)

# Data Cleaning: Remove rows with errors in rating or missing values
movies_df = movies_df[~movies_df['rating'].str.contains('min', na=False)]
movies_df = movies_df.dropna(subset=['duration_min', 'rating', 'release_year'])

# 3. Define Features (X) and Target (y)
X = movies_df[['release_year', 'rating']]
y = movies_df['duration_min']

# 4. Build Model Pipeline
# OneHotEncoder converts text 'rating' into numbers for the math model
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['rating'])
    ], remainder='passthrough')

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# 6. Save Model and Metadata
joblib.dump(model_pipeline, 'netflix_duration_model.pkl')
unique_ratings = sorted(movies_df['rating'].unique().tolist())
with open('ratings.txt', 'w') as f:
    for r in unique_ratings:
        f.write(f"{r}\n")

print("Model trained and saved as 'netflix_duration_model.pkl'")