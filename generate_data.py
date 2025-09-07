import pandas as pd
from faker import Faker
import numpy as np
import random

# Initialize Faker to generate fake data
fake = Faker()

# Number of customers
NUM_CUSTOMERS = 500

# Industries list
industries = ['Tech', 'Healthcare', 'Finance', 'Retail', 'Manufacturing']

# Generate data
data = []
for i in range(NUM_CUSTOMERS):
    engagement_score = random.randint(1, 100)
    total_purchases = random.randint(1, 50) * 1000
    last_interaction_days = random.randint(1, 365)
    
    # Simple logic for churn based on engagement and interaction
    churn_probability = (100 - engagement_score) / 100 * (last_interaction_days / 365)
    churn = 1 if churn_probability > 0.6 and random.random() > 0.2 else 0

    data.append({
        'CustomerID': 1001 + i,
        'CompanyName': fake.company(),
        'Industry': random.choice(industries),
        'TotalPurchases': total_purchases,
        'LastInteractionDaysAgo': last_interaction_days,
        'EngagementScore': engagement_score,
        'Churn': churn
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('mock_crm_data.csv', index=False)

print(f"Successfully generated mock_crm_data.csv with {NUM_CUSTOMERS} customers.")