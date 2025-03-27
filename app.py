import streamlit as st
import docx
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os

# Download NLTK resources only if not already downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)

# Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text])

def preprocess_resume_text(text):
    """
    Preprocess resume text to match training data format
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and lemmatize
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

@st.cache_resource
def load_models():
    """Load all required models with caching"""
    try:
        # Load TF-IDF vectorizer
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        
        # Load Gradient Boosting model
        model = joblib.load('gradient_boosting.pkl')
        
        # Initialize LabelEncoder (assuming encoder.py contains the class)
        le = joblib.load('encoder.pkl')
        
        return tfidf, model, le
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def main():
    st.title("Resume Classification System")
    st.markdown("Upload a resume in DOCX format to classify its category")
    
    # Load models
    tfidf, model, le = load_models()
    
    if tfidf is None or model is None or le is None:
        st.error("Failed to load required models. Please check your model files.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a DOCX resume file", type="docx")
    
    if uploaded_file is not None:
        # Extract and preprocess text
        raw_text = extract_text_from_docx(uploaded_file)
        processed_text = preprocess_resume_text(raw_text)
        
        # Display sections
        with st.expander("View Extracted Resume Text"):
            st.text(raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""))
        
        with st.expander("View Processed Text"):
            st.text(processed_text[:2000] + ("..." if len(processed_text) > 2000 else ""))
        
        # Transform and predict
        text_tfidf = tfidf.transform([processed_text])
        prediction = model.predict(text_tfidf)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_tfidf)[0]
            top_classes = np.argsort(probabilities)[::-1][:5]  # Top 5 predictions
            
            st.subheader("Classification Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Category", le.inverse_transform(prediction)[0])
            
            with col2:
                st.metric("Confidence Score", f"{max(probabilities)*100:.1f}%")
            
            st.subheader("Top 5 Possible Categories")
            for i, class_idx in enumerate(top_classes):
                prob = probabilities[class_idx]
                st.progress(prob, text=f"{le.inverse_transform([class_idx])[0]}: {prob*100:.1f}%")
        else:
            st.success(f"Predicted Category: {le.inverse_transform(prediction)[0]}")

if __name__ == "__main__":
    main()