import streamlit as st
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word files
from sklearn.decomposition import PCA

def preprocess(text):
    if not text.strip():  # Handle empty input
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = text.split()  # Tokenize text into words
    stop_words = set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 
        'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
        'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
        'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 
        'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 
        'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    ])  # Define stopwords list
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    
    # Ensure that the resulting string is non-empty after filtering
    if not filtered_tokens:
        return ""
    
    return " ".join(filtered_tokens)

def read_file(file):
    file_type = file.name.split('.')[-1].lower()  # Get the file extension
    
    if file_type == 'txt':
        return file.read().decode('utf-8')  # Read text file
    
    elif file_type == 'pdf':
        # Read PDF content using PyMuPDF
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    
    elif file_type == 'docx':
        # Read Word content using python-docx
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    else:
        return ""  # If file is not supported

# Streamlit interface
st.title("Document Clustering App")

# File uploader (allowing multiple file uploads)
uploaded_files = st.file_uploader("Upload Files", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    # Initialize lists to store documents and filenames
    documents = []
    filenames = []

    for uploaded_file in uploaded_files:
        content = read_file(uploaded_file)  # Read content based on file type
        preprocessed_text = preprocess(content)  # Preprocess text
        if preprocessed_text.strip():  # Check if the text is not empty
            documents.append(preprocessed_text)
            filenames.append(uploaded_file.name)

    # Check if documents are empty after preprocessing
    if not documents:
        st.error("Error: All documents are empty after preprocessing!")
    else:
        # Filter out empty or too short documents (less than 5 words after preprocessing)
        documents = [doc for doc in documents if len(doc.split()) > 0]
        filenames = [filename for idx, filename in enumerate(filenames) if len(documents[idx].split()) > 0]

        if not documents:
            st.error("Error: All documents are too short after preprocessing!")
        else:
            # Proceed with TF-IDF and KMeans clustering if there are valid documents
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(documents)

            # Dynamically set the number of clusters based on the number of documents
            n_clusters = min(3, len(documents))  # Set n_clusters to 3 or the number of documents, whichever is smaller

            # Check if there's only one document
            if n_clusters == 1:
                st.warning("Only one document found, clustering can't be performed. Displaying the document.")
            else:
                # KMeans clustering
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(X)

                # Display results in a table
                result_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
                st.write(result_df)

                # Optional: Plotting using Plotly (2D PCA Plot)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X.toarray())

                fig = px.scatter(
                    x=X_pca[:, 0], 
                    y=X_pca[:, 1], 
                    color=labels.astype(str),
                    hover_name=filenames,
                    title="📊 Document Clusters (PCA 2D Plot)"
                )
                st.plotly_chart(fig)
