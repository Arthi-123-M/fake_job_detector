"""
Fake Job Detection - Streamlit Web App
This app provides an interactive interface for fake job detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .fake-job {
        background-color: #ff6b6b;
        color: white;
    }
    .real-job {
        background-color: #51cf66;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Header
st.markdown("""
    <div class="main-header">
        <h1>🔍 Fake Job Detection System</h1>
        <p>Detect fraudulent job postings using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/job.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 Model Performance", "🔮 Predict New Job", "📈 Dataset Analysis", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.info("Using Kaggle Fake Job Posting Dataset (17,880 jobs)")
    
    st.markdown("### 🤖 Models Used")
    st.success("• Logistic Regression")
    st.success("• Random Forest")

# Load and preprocess data function
@st.cache_data
def load_and_preprocess():
    # Check if dataset exists
    if os.path.exists('fake_job_postings.csv'):
        try:
            df = pd.read_csv('fake_job_postings.csv')
            st.success("✅ Dataset loaded successfully!")
            return df
        except Exception as e:
            st.warning(f"Error loading dataset: {e}")
            return create_sample_data()
    else:
        st.warning("📌 Dataset not found. Using sample data for demonstration.")
        st.info("To use the full dataset, upload 'fake_job_postings.csv' to your repository.")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic job titles
    real_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 
                   'Marketing Specialist', 'Sales Representative', 'Financial Analyst']
    fake_titles = ['URGENT HIRING!!!', 'Work From Home - Earn $5000', 
                   'Get Rich Quick!!!', 'Easy Money Online', 'No Experience Needed!!!']
    
    titles = real_titles + fake_titles
    
    data = {
        'title': np.random.choice(titles, n_samples),
        'description': np.random.choice([
            'We are looking for an experienced professional to join our team...',
            'EARN BIG MONEY FAST!!! No experience needed. Start today!',
            'Join our growing company with excellent benefits...',
            'Limited time opportunity! Work from home and make thousands!'
        ], n_samples),
        'requirements': np.random.choice([
            '5+ years experience, Bachelor\'s degree required',
            'No experience required, just internet connection',
            'Python, SQL, and Machine Learning skills',
            'Must have reliable internet connection'
        ], n_samples),
        'company_profile': np.random.choice([
            'Established company with 50+ years of excellence',
            '',
            'Fortune 500 company with offices worldwide',
            ''
        ], n_samples),
        'fraudulent': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    return pd.DataFrame(data)

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Train models function
@st.cache_resource
def train_models(df):
    # Preprocess
    df = df.fillna(' ')
    
    # Clean text
    df['title'] = df['title'].apply(clean_text)
    df['description'] = df['description'].apply(clean_text)
    df['requirements'] = df['requirements'].apply(clean_text)
    
    # Combine text
    df['all_text'] = df['title'] + ' ' + df['description'] + ' ' + df['requirements']
    
    # Feature engineering
    df['text_length'] = df['all_text'].apply(len)
    df['desc_length'] = df['description'].apply(len)
    
    # Suspicious words
    suspicious_words = ['urgent', 'immediate', 'work from home', 'earn money', 
                       'quick cash', 'no experience', 'guaranteed', 'get rich',
                       'easy money', 'fast cash', 'millionaire']
    
    def count_suspicious(text):
        return sum(1 for word in suspicious_words if word in text)
    
    df['suspicious_count'] = df['all_text'].apply(count_suspicious)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_text = vectorizer.fit_transform(df['all_text'])
    
    # Numeric features
    numeric_features = ['text_length', 'desc_length', 'suspicious_count']
    X_numeric = df[numeric_features].values
    
    # Combine features
    from scipy.sparse import hstack
    X = hstack([X_text, X_numeric])
    y = df['fraudulent'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    
    with st.spinner("Training Logistic Regression..."):
        lr_model.fit(X_train, y_train)
    
    with st.spinner("Training Random Forest..."):
        rf_model.fit(X_train, y_train)
    
    # Make predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    
    return {
        'vectorizer': vectorizer,
        'lr_model': lr_model,
        'rf_model': rf_model,
        'X_test': X_test,
        'y_test': y_test,
        'lr_pred': lr_pred,
        'rf_pred': rf_pred,
        'lr_proba': lr_proba,
        'rf_proba': rf_proba,
        'numeric_features': numeric_features,
        'suspicious_words': suspicious_words
    }

# Load data and train models
with st.spinner("Loading data and training models..."):
    df = load_and_preprocess()
    if not st.session_state.model_trained:
        models = train_models(df)
        st.session_state.model_trained = True
        st.session_state.models = models
        st.session_state.df = df

# Different pages
if page == "🏠 Home":
    st.markdown("## 📊 Welcome to the Fake Job Detection System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fake_count = (st.session_state.df['fraudulent'] == 1).sum()
        real_count = (st.session_state.df['fraudulent'] == 0).sum()
        total = len(st.session_state.df)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>📁 Dataset Size</h3>
            <h2>{total:,}</h2>
            <p>Job Postings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fake_pct = (fake_count / total) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Fake Jobs</h3>
            <h2>{fake_pct:.1f}%</h2>
            <p>{fake_count:,} postings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Models</h3>
            <h2>2</h2>
            <p>Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Key Features
    - **Dual Algorithm Approach**: Compare Logistic Regression and Random Forest
    - **Real-time Prediction**: Test any job posting instantly
    - **Comprehensive Analysis**: View model performance metrics
    - **Interactive Visualizations**: Explore data and results
    
    ### 🔍 How to Use
    1. Go to **Predict New Job** to test a job posting
    2. Check **Model Performance** to see accuracy metrics
    3. Explore **Dataset Analysis** for insights
    """)

# ... (rest of your pages code remains the same)
