import os

@st.cache_data
def load_and_preprocess():
    # Try sample dataset first (under 25MB)
    if os.path.exists('sample_job_postings.csv'):
        try:
            df = pd.read_csv('sample_job_postings.csv')
            st.success(f"✅ Using sample dataset with {len(df):,} job postings!")
            return df
        except Exception as e:
            st.warning(f"Error loading sample dataset: {e}")
    
    # Try full dataset if available
    elif os.path.exists('fake_job_postings.csv'):
        try:
            df = pd.read_csv('fake_job_postings.csv')
            st.success(f"✅ Using full dataset with {len(df):,} job postings!")
            return df
        except Exception as e:
            st.warning(f"Error loading dataset: {e}")
    
    # Fall back to sample data
    st.warning("📌 Using built-in sample data for demonstration")
    return create_sample_data()
