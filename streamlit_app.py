"""
Fake Job Detection - Streamlit Web App
"""

# First, import all required libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Page config MUST come after imports but before any other st commands
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
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Header
st.markdown("""
    <div class="main-header">
        <h1>🔍 Fake Job Detection System</h1>
        <p>Detect fraudulent job postings using Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Create sample data function (must be defined before caching)
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
    
    # Create descriptions
    real_descriptions = [
        'We are looking for an experienced professional to join our team. Competitive salary and benefits package offered.',
        'Join our growing company with excellent career growth opportunities. Great work environment.',
        'Seeking a talented individual to contribute to our innovative projects. Full-time position with benefits.'
    ]
    
    fake_descriptions = [
        'EARN BIG MONEY FAST!!! No experience needed. Start today! Limited positions!',
        'Work from home and make thousands! Guaranteed income! Immediate start!',
        'Get rich quick! No investment required! Start earning immediately!'
    ]
    
    # Create requirements
    real_requirements = [
        '5+ years experience, Bachelor\'s degree required',
        'Python, SQL, and Machine Learning skills',
        'Strong communication skills, team player'
    ]
    
    fake_requirements = [
        'No experience required, just internet connection',
        'Must have reliable internet connection, work from home',
        'None, anyone can do this job'
    ]
    
    data = {
        'title': np.random.choice(titles, n_samples),
        'description': np.random.choice(real_descriptions + fake_descriptions, n_samples),
        'requirements': np.random.choice(real_requirements + fake_requirements, n_samples),
        'company_profile': np.random.choice([
            'Established company with 50+ years of excellence',
            'Fortune 500 company with offices worldwide',
            'Startup with great potential',
            '',
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

# Load and preprocess data function
@st.cache_data
def load_and_preprocess():
    """Load dataset from various sources"""
    
    # Try sample dataset first (if uploaded)
    if os.path.exists('sample_job_postings.csv'):
        try:
            df = pd.read_csv('sample_job_postings.csv')
            st.success(f"✅ Using sample dataset with {len(df):,} job postings!")
            return df
        except Exception as e:
            st.warning(f"Could not load sample dataset: {e}")
    
    # Try full dataset if available
    elif os.path.exists('fake_job_postings.csv'):
        try:
            df = pd.read_csv('fake_job_postings.csv')
            st.success(f"✅ Using full dataset with {len(df):,} job postings!")
            return df
        except Exception as e:
            st.warning(f"Could not load full dataset: {e}")
    
    # Fall back to generated sample data
    st.info("📌 Using built-in sample data for demonstration")
    return create_sample_data()

# Train models function
@st.cache_resource
def train_models(df):
    """Train machine learning models"""
    
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
    if not st.session_state.models_trained:
        models = train_models(df)
        st.session_state.models_trained = True
        st.session_state.models = models
        st.session_state.df = df

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/job.png", width=100)
    st.title("Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
    
    if st.button("📊 Model Performance", use_container_width=True):
        st.session_state.page = "Model Performance"
        st.rerun()
    
    if st.button("🔮 Predict New Job", use_container_width=True):
        st.session_state.page = "Predict New Job"
        st.rerun()
    
    if st.button("📈 Dataset Analysis", use_container_width=True):
        st.session_state.page = "Dataset Analysis"
        st.rerun()
    
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "About"
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.info(f"Using dataset with {len(st.session_state.df):,} jobs")
    
    st.markdown("### 🤖 Models Used")
    st.success("• Logistic Regression")
    st.success("• Random Forest")

# Page content
if st.session_state.page == "Home":
    st.title("🏠 Welcome to Fake Job Detection System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_jobs = len(st.session_state.df)
        st.metric("Total Jobs", f"{total_jobs:,}")
    
    with col2:
        fake_count = (st.session_state.df['fraudulent'] == 1).sum()
        fake_pct = (fake_count / total_jobs) * 100
        st.metric("Fake Jobs", f"{fake_count:,} ({fake_pct:.1f}%)")
    
    with col3:
        real_count = (st.session_state.df['fraudulent'] == 0).sum()
        real_pct = (real_count / total_jobs) * 100
        st.metric("Real Jobs", f"{real_count:,} ({real_pct:.1f}%)")
    
    st.markdown("---")
    st.markdown("""
    ### 🎯 How to Use:
    1. Click **"Predict New Job"** in the sidebar to test a job posting
    2. Fill in the job details
    3. See instant predictions from both models
    
    ### 🔍 What We Detect:
    - Urgent/Immediate hiring language
    - "Work from home" promises with no experience
    - Money-focused descriptions
    - Missing company information
    - Vague requirements
    """)

elif st.session_state.page == "Predict New Job":
    st.title("🔮 Test a Job Posting")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Job Title*", value="Senior Python Developer")
            company = st.text_input("Company Name", value="Tech Solutions Inc.")
            telecommuting = st.checkbox("Telecommuting / Remote")
        
        with col2:
            description = st.text_area("Job Description*", 
                value="We are looking for an experienced Python developer with 5+ years of experience.",
                height=150)
            requirements = st.text_area("Requirements*",
                value="Python, Django, SQL, AWS",
                height=100)
        
        submitted = st.form_submit_button("🔍 Detect Fake Job", use_container_width=True)
    
    if submitted:
        # Combine text
        all_text = f"{title} {description} {requirements}"
        all_text = clean_text(all_text)
        
        # Vectorize
        text_vector = st.session_state.models['vectorizer'].transform([all_text])
        
        # Calculate features
        text_length = len(all_text)
        desc_length = len(description)
        
        suspicious_count = 0
        for word in st.session_state.models['suspicious_words']:
            if word in all_text:
                suspicious_count += 1
        
        # Create numeric array
        numeric_array = np.array([[text_length, desc_length, suspicious_count]])
        
        # Combine features
        from scipy.sparse import hstack
        features = hstack([text_vector, numeric_array])
        
        # Get predictions
        lr_pred = st.session_state.models['lr_model'].predict(features)[0]
        rf_pred = st.session_state.models['rf_model'].predict(features)[0]
        
        lr_proba = st.session_state.models['lr_model'].predict_proba(features)[0]
        rf_proba = st.session_state.models['rf_model'].predict_proba(features)[0]
        
        # Display results
        st.markdown("### 📊 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Logistic Regression")
            if lr_pred == 1:
                st.markdown("""
                <div class="prediction-box fake-job">
                    🔴 FAKE JOB POSTING
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box real-job">
                    🟢 REAL JOB POSTING
                </div>
                """, unsafe_allow_html=True)
            
            confidence = max(lr_proba) * 100
            st.progress(float(max(lr_proba)))
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.write(f"Fake: {lr_proba[1]*100:.1f}% | Real: {lr_proba[0]*100:.1f}%")
        
        with col2:
            st.markdown("#### Random Forest")
            if rf_pred == 1:
                st.markdown("""
                <div class="prediction-box fake-job">
                    🔴 FAKE JOB POSTING
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box real-job">
                    🟢 REAL JOB POSTING
                </div>
                """, unsafe_allow_html=True)
            
            confidence = max(rf_proba) * 100
            st.progress(float(max(rf_proba)))
            st.write(f"**Confidence:** {confidence:.1f}%")
            st.write(f"Fake: {rf_proba[1]*100:.1f}% | Real: {rf_proba[0]*100:.1f}%")

elif st.session_state.page == "Model Performance":
    st.title("📊 Model Performance")
    st.info("Model performance metrics will be displayed here with confusion matrices and ROC curves.")
    
    # You can add performance visualizations here

elif st.session_state.page == "Dataset Analysis":
    st.title("📈 Dataset Analysis")
    
    # Show basic dataset info
    st.write(f"**Total samples:** {len(st.session_state.df)}")
    st.write(f"**Features:** {len(st.session_state.df.columns)}")
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(st.session_state.df.head(10))

elif st.session_state.page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    ### Fake Job Detection System
    
    This application uses Machine Learning to detect fraudulent job postings.
    
    **Features:**
    - Real-time job posting analysis
    - Dual algorithm comparison (Logistic Regression & Random Forest)
    - Confidence scores for predictions
    - Easy-to-use interface
    
    **Technologies Used:**
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - NumPy
    
    **Developer:** Arthi M
    
    **Dataset:** Kaggle Fake Job Posting Dataset
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center'>Made with Streamlit</div>", unsafe_allow_html=True)
