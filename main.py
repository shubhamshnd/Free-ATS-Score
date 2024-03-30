import streamlit as st
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load('en_core_web_sm')

def extract_keywords(text):
    doc = nlp(text)
    # Extract nouns and adjectives as keywords
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]
    return keywords

def calculate_similarity(resume_text, job_description_text):
    corpus = [resume_text, job_description_text]
    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

def calculate_ats_score(similarity_score):
    if similarity_score >= 0.9:
        return "Excellent Match"
    elif similarity_score >= 0.8:
        return "Good Match"
    elif similarity_score >= 0.7:
        return "Fair Match"
    else:
        return "Poor Match"

def main():
    st.title('Resume Matching Tool')

    st.sidebar.header('Upload Files')
    resume_file = st.sidebar.file_uploader('Upload your Resume', type=['pdf'])
    job_description = st.sidebar.text_area("Job Description", "")

    if resume_file and job_description:
        resume_text = resume_file.read()

        # Display uploaded files
        st.subheader('Uploaded Files')
        st.write('Resume:', resume_file.name)

        # Extract keywords from job description
        keywords = extract_keywords(job_description)
        st.subheader('Keywords extracted from Job Description')
        st.write(keywords)

        # Calculate similarity
        similarity_score = calculate_similarity(resume_text, job_description)
        st.subheader('Matching Score')
        st.write(f'The matching score between your resume and the job description is: {similarity_score:.2%}')

        # Calculate ATS score
        ats_score = calculate_ats_score(similarity_score)
        st.subheader('ATS Score')
        st.write(f'Your ATS score based on the matching score is: **{ats_score}**', unsafe_allow_html=True)

if __name__ == '__main__':
    main()