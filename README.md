

**Instructions:**

1.  In your project folder, create a new file named `README.md`.
2.  Copy the entire content below and paste it into that file.
3.  **Crucially, take a screenshot of your running application and replace the placeholder text with the actual image.**

-----

````markdown
# 🧠 AI-Enhanced CRM Insights Generator

**Submission for the Mini Hackathon (Use Case: PS-1)**

This project is an interactive web application designed to provide a B2B sales team with actionable insights from their customer data. It uses machine learning to segment customers, predict churn, and features an AI-powered assistant for strategic querying.

---

## 📸 Application Preview


*A screenshot of the main dashboard showing the customer segmentation chart and key metrics.*

---

## ✨ Key Features

* **Interactive Dashboard:** Built with Streamlit to provide a clean and responsive user interface for exploring customer data.
* **Automated Customer Segmentation:** Utilizes a K-Means clustering algorithm to group customers into four key segments: `High-Value`, `Loyal`, `At-Risk`, and `New`.
* **Predictive Churn Analysis:** Employs a Random Forest model to calculate the churn probability for each customer, allowing for proactive retention efforts.
* **AI-Powered Assistant:** Integrated with the Google Gemini API, this chatbot can answer strategic, natural-language questions about the customer data, providing instant, actionable insights.
* **Dynamic Visualizations:** Uses Plotly Express to create interactive charts and tables for intuitive data exploration.

---

## 🛠️ Tech Stack

* **Backend & ML:** Python
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-learn
* **Visualizations:** Plotly Express
* **Generative AI:** Google Gemini API

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* Python 3.9+
* A Google Gemini API Key

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/crm-hackathon-project.git](https://github.com/your-username/crm-hackathon-project.git)
cd crm-hackathon-project
````

### 2\. Create and Activate a Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4\. Configure Your API Key

Create a secret file to store your Gemini API Key securely.

  * Create a folder: `.streamlit`
  * Inside it, create a file: `secrets.toml`
  * Add your key to the file in the following format:

<!-- end list -->

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "YOUR_SECRET_API_KEY_GOES_HERE"
```

### 5\. Prepare the Data and Models

Run the following scripts in order to generate the mock dataset and train the machine learning models.

```bash
# 1. Generate mock_crm_data.csv
python generate_data.py

# 2. Train and save churn_model.joblib & segmentation_model.joblib
python train.py
```

### 6\. Run the Application

You are now ready to launch the Streamlit app\!

```bash
streamlit run app.py
```

The application will open in your web browser.

-----

## 📂 Project Structure

```
.
├── .streamlit/
│   └── secrets.toml      # Stores the API key
├── app.py                # Main Streamlit application file
├── generate_data.py      # Script to create mock CRM data
├── train.py              # Script to train and save ML models
├── mock_crm_data.csv     # The generated dataset
├── churn_model.joblib    # Saved churn prediction model
├── segmentation_model.joblib # Saved segmentation model
├── requirements.txt      # Project dependencies
└── README.md             # You are here!
```

```
```
