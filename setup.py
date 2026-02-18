rom setuptools import setup, find_packages

setup(
    name="fake-job-detector",
    version="1.0.0",
    description="A machine learning system to detect fraudulent job postings",
    author="Arthi M",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "scipy>=1.11.0"
    ],
    python_requires=">=3.8",
)
