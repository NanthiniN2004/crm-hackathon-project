import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import joblib

print("Starting model training...")

# Load data
df = pd.read_csv('mock_crm_data.csv')

# --- Define features for each model ---
# Features for churn prediction
churn_features = ['TotalPurchases', 'LastInteractionDaysAgo', 'EngagementScore', 'Industry']
churn_target = 'Churn'

# Features for customer segmentation (clustering)
segmentation_features = ['TotalPurchases', 'EngagementScore']

# --- Preprocessing ---
# Create a preprocessor for numeric and categorical features
numeric_features = ['TotalPurchases', 'LastInteractionDaysAgo', 'EngagementScore']
categorical_features = ['Industry']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Churn Model Training ---
print("Training churn prediction model...")
# Create the full pipeline with preprocessor and classifier
churn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Separate data for churn model
X_churn = df[churn_features]
y_churn = df[churn_target]

# Train the model
churn_pipeline.fit(X_churn, y_churn)

# Save the churn model pipeline
joblib.dump(churn_pipeline, 'churn_model.joblib')
print("Churn model saved as churn_model.joblib")

# --- Segmentation Model Training ---
print("Training customer segmentation model...")
# For segmentation, we only use numeric features and scale them
segmentation_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10)) # Added n_init explicitly
])

# Select and train on segmentation data
X_segmentation = df[segmentation_features]
segmentation_pipeline.fit(X_segmentation)

# Save the segmentation model pipeline
joblib.dump(segmentation_pipeline, 'segmentation_model.joblib')
print("Segmentation model saved as segmentation_model.joblib")

print("\nModel training complete!")