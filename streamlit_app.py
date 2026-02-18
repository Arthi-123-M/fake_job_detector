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
    try:
        df = pd.read_csv('fake_job_postings.csv')
        
        # Convert target: 1 = FAKE (original), 0 = REAL (original)
        # We'll keep original mapping for clarity
        return df
    except:
        # Create sample data if file not found
        st.warning("Dataset not found. Using sample data for demonstration.")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'title': np.random.choice(['Software Engineer', 'Data Scientist', 'Marketing Manager', 
                                   'URGENT HIRING', 'Work From Home', 'Get Rich Quick'], n_samples),
        'description': np.random.choice(['Looking for experienced professional', 
                                         'EARN BIG MONEY FAST!!! No experience needed'], n_samples),
        'requirements': np.random.choice(['Python, SQL, Degree required', 
                                          'No experience required, just internet'], n_samples),
        'company_profile': np.random.choice(['Established company with 50+ years', ''], n_samples),
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
                       'quick cash', 'no experience', 'guaranteed']
    
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
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    
    lr_model.fit(X_train, y_train)
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
        'numeric_features': numeric_features
    }

# Load data and train models
with st.spinner("Loading data and training models..."):
    df = load_and_preprocess()
    if not st.session_state.model_trained:
        models = train_models(df)
        st.session_state.model_trained = True
        st.session_state.models = models

# Different pages
if page == "🏠 Home":
    st.markdown("## 📊 Welcome to the Fake Job Detection System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📁 Dataset Size</h3>
            <h2>{:,}</h2>
            <p>Job Postings</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        fake_pct = (df['fraudulent'].sum() / len(df)) * 100
        st.markdown("""
        <div class="metric-card">
            <h3>⚠️ Fake Jobs</h3>
            <h2>{:.1f}%</h2>
            <p>of dataset</p>
        </div>
        """.format(fake_pct), unsafe_allow_html=True)
    
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

elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Comparison")
    
    models = st.session_state.models
    
    # Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Logistic Regression")
        lr_cm = confusion_matrix(models['y_test'], models['lr_pred'])
        
        fig_lr = go.Figure(data=go.Heatmap(
            z=lr_cm,
            x=['Predicted Fake', 'Predicted Real'],
            y=['Actual Fake', 'Actual Real'],
            text=lr_cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        ))
        fig_lr.update_layout(title='Confusion Matrix - Logistic Regression')
        st.plotly_chart(fig_lr, use_container_width=True)
        
        lr_acc = (lr_cm[0][0] + lr_cm[1][1]) / lr_cm.sum()
        st.metric("Accuracy", f"{lr_acc:.2%}")
    
    with col2:
        st.markdown("### Random Forest")
        rf_cm = confusion_matrix(models['y_test'], models['rf_pred'])
        
        fig_rf = go.Figure(data=go.Heatmap(
            z=rf_cm,
            x=['Predicted Fake', 'Predicted Real'],
            y=['Actual Fake', 'Actual Real'],
            text=rf_cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Greens'
        ))
        fig_rf.update_layout(title='Confusion Matrix - Random Forest')
        st.plotly_chart(fig_rf, use_container_width=True)
        
        rf_acc = (rf_cm[0][0] + rf_cm[1][1]) / rf_cm.sum()
        st.metric("Accuracy", f"{rf_acc:.2%}")
    
    # ROC Curves
    st.markdown("### ROC Curves")
    
    # Calculate ROC curves
    lr_fpr, lr_tpr, _ = roc_curve(models['y_test'], models['lr_proba'][:, 1])
    rf_fpr, rf_tpr, _ = roc_curve(models['y_test'], models['rf_proba'][:, 1])
    
    lr_auc = auc(lr_fpr, lr_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=lr_fpr, y=lr_tpr, mode='lines', 
                                 name=f'Logistic Regression (AUC={lr_auc:.3f})',
                                 line=dict(color='blue', width=2)))
    fig_roc.add_trace(go.Scatter(x=rf_fpr, y=rf_tpr, mode='lines',
                                 name=f'Random Forest (AUC={rf_auc:.3f})',
                                 line=dict(color='green', width=2)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random', line=dict(color='red', dash='dash')))
    
    fig_roc.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        title='ROC Curves Comparison'
    )
    st.plotly_chart(fig_roc, use_container_width=True)

elif page == "🔮 Predict New Job":
    st.markdown("## 🔮 Test a Job Posting")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Job Title", value="Senior Python Developer")
            company = st.text_input("Company Name", value="Tech Solutions Inc.")
            telecommuting = st.checkbox("Telecommuting / Remote")
        
        with col2:
            description = st.text_area("Job Description", 
                value="We are looking for a senior Python developer with 5+ years of experience.",
                height=150)
            requirements = st.text_area("Requirements",
                value="Python, Django, REST APIs, AWS, SQL",
                height=100)
        
        submitted = st.form_submit_button("🔍 Detect Fake Job")
    
    if submitted:
        models = st.session_state.models
        
        # Prepare text
        all_text = f"{title} {description} {requirements}"
        all_text = clean_text(all_text)
        
        # Vectorize
        text_vector = models['vectorizer'].transform([all_text])
        
        # Calculate features
        text_length = len(all_text)
        desc_length = len(description)
        
        suspicious_words = ['urgent', 'immediate', 'work from home', 'earn money', 
                           'quick cash', 'no experience', 'guaranteed']
        suspicious_count = sum(1 for word in suspicious_words if word in all_text)
        
        # Create numeric array
        numeric_array = np.array([[text_length, desc_length, suspicious_count]])
        
        # Combine features
        from scipy.sparse import hstack
        features = hstack([text_vector, numeric_array])
        
        # Get predictions
        lr_pred = models['lr_model'].predict(features)[0]
        rf_pred = models['rf_model'].predict(features)[0]
        
        lr_proba = models['lr_model'].predict_proba(features)[0]
        rf_proba = models['rf_model'].predict_proba(features)[0]
        
        # Display results
        st.markdown("### Prediction Results")
        
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
            
            st.progress(float(lr_proba[1]))
            st.text(f"Confidence: {max(lr_proba)*100:.1f}%")
            st.text(f"Fake: {lr_proba[0]*100:.1f}% | Real: {lr_proba[1]*100:.1f}%")
        
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
            
            st.progress(float(rf_proba[1]))
            st.text(f"Confidence: {max(rf_proba)*100:.1f}%")
            st.text(f"Fake: {rf_proba[0]*100:.1f}% | Real: {rf_proba[1]*100:.1f}%")

elif page == "📈 Dataset Analysis":
    st.markdown("## 📈 Dataset Analysis")
    
    # Class distribution
    st.markdown("### Class Distribution")
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Real Jobs', 'Fake Jobs'],
        values=[(df['fraudulent'] == 0).sum(), (df['fraudulent'] == 1).sum()],
        hole=.3,
        marker_colors=['#51cf66', '#ff6b6b']
    )])
    fig_pie.update_layout(title='Distribution of Fake vs Real Jobs')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Sample data
    st.markdown("### Sample Data")
    st.dataframe(df.head(10))

else:  # About page
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This Fake Job Detection System uses machine learning to identify fraudulent job postings.
    It's designed to help job seekers avoid scams and recruiters maintain quality listings.
    
    ### 🤖 Algorithms Used
    
    #### 1. Logistic Regression
    - **Output**: 0 (FAKE) or 1 (REAL)
    - **Strengths**: Fast, interpretable, good for baseline
    - **Best for**: Quick screening
    
    #### 2. Random Forest
    - **Output**: 🔴 RED (FAKE) or 🟢 GREEN (REAL)
    - **Strengths**: Handles complex patterns, feature importance
    - **Best for**: Accurate detection
    
    ### 🔍 Key Features Detected
    - 🚨 Urgency words (urgent, immediate)
    - 💰 Money-focused language
    - 📝 Missing company profiles
    - 🔗 Suspicious patterns
    
    ### 📊 Dataset
    - **Source**: Kaggle Fake Job Posting Dataset
    - **Size**: 17,880 job postings
    - **Features**: Title, description, requirements, company info
    
    ### 🛠️ Technologies Used
    - Python 3.8+
    - Scikit-learn for ML models
    - Streamlit for web interface
    - Plotly for visualizations
    - Pandas for data manipulation
    
    ### 📧 Contact
    **Developer**: Arthi M
    **GitHub**: [Arthi-123-M](https://github.com/Arthi-123-M)
    
    ### 📝 License
    This project is licensed under the MIT License.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made with ❤️ using Streamlit | Fake Job Detection System v1.0</p>
    </div>
""", unsafe_allow_html=True)