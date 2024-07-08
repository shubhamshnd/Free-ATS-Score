import streamlit as st
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.getvalue()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop]
    return list(set(keywords))  # Remove duplicates

def calculate_similarity(resume_text, job_description_text):
    corpus = [resume_text, job_description_text]
    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def calculate_ats_score(similarity_score):
    if similarity_score >= 0.8:
        return "Excellent Match", 5
    elif similarity_score >= 0.6:
        return "Good Match", 4
    elif similarity_score >= 0.4:
        return "Fair Match", 3
    elif similarity_score >= 0.2:
        return "Poor Match", 2
    else:
        return "Very Poor Match", 1

def get_missing_keywords(resume_keywords, job_keywords):
    return list(set(job_keywords) - set(resume_keywords))

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

def generate_skill_radar_chart(resume_skills, job_skills):
    try:
        skills = list(set(resume_skills + job_skills))
        
        # Limit the number of skills to display
        max_skills = 15
        if len(skills) > max_skills:
            # Sort skills by their presence in both resume and job description
            skill_importance = {skill: (skill in resume_skills) + (skill in job_skills) for skill in skills}
            skills = sorted(skill_importance, key=skill_importance.get, reverse=True)[:max_skills]
        
        resume_values = [1 if skill in resume_skills else 0 for skill in skills]
        job_values = [1 if skill in job_skills else 0 for skill in skills]
        
        angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False)
        
        # Close the plot
        resume_values += resume_values[:1]
        job_values += job_values[:1]
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, resume_values, 'o-', linewidth=2, label='Resume')
        ax.fill(angles, resume_values, alpha=0.25)
        ax.plot(angles, job_values, 'o-', linewidth=2, label='Job Description')
        ax.fill(angles, job_values, alpha=0.25)
        
        ax.set_thetagrids(angles[:-1] * 180/np.pi, skills)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        return fig
    except Exception as e:
        st.error(f"An error occurred while generating the skill radar chart: {str(e)}")
        return None

def generate_keyword_frequency_chart(resume_text, job_text):
    resume_words = resume_text.lower().split()
    job_words = job_text.lower().split()
    word_freq = {}
    
    for word in set(resume_words + job_words):
        word_freq[word] = [resume_words.count(word), job_words.count(word)]
    
    df = pd.DataFrame.from_dict(word_freq, orient='index', columns=['Resume', 'Job Description'])
    df = df.sort_values(by=['Job Description', 'Resume'], ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    plt.title('Top 10 Keyword Frequencies')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    return fig

def main():
    st.set_page_config(layout="wide")  # Use wide layout for better space utilization
    st.title('Advanced Resume Matching Tool')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.sidebar.header('Upload Files')
        resume_file = st.sidebar.file_uploader('Upload your Resume (PDF)', type=['pdf'])
        job_description = st.sidebar.text_area("Job Description", "")
        analyze_button = st.sidebar.button('Start Analysis')

    if resume_file and job_description and analyze_button:
        try:
            resume_text = extract_text_from_pdf(resume_file)

            with col2:
                st.subheader('Uploaded Files')
                st.write('Resume:', resume_file.name)

                # Extract keywords
                resume_keywords = extract_keywords(resume_text)
                job_keywords = extract_keywords(job_description)

                # Calculate similarity
                similarity_score = calculate_similarity(resume_text, job_description)
                st.subheader('Matching Score')
                st.write(f'The matching score between your resume and the job description is: {similarity_score:.2%}')

                # Calculate ATS score
                ats_score, score_value = calculate_ats_score(similarity_score)
                st.subheader('ATS Score')
                st.write(f'Your ATS score: **{ats_score}** ({score_value}/5)', unsafe_allow_html=True)

                # Progress bar for visual representation
                st.progress(score_value / 5)

            # Keyword Analysis
            st.subheader('Keyword Analysis')
            keyword_col1, keyword_col2 = st.columns(2)
            with keyword_col1:
                st.write('Resume Keywords')
                st.write(', '.join(resume_keywords))
            with keyword_col2:
                st.write('Job Description Keywords')
                st.write(', '.join(job_keywords))

            # Missing keywords
            st.subheader('Missing Keywords')
            if missing_keywords := get_missing_keywords(resume_keywords, job_keywords):
                st.write("Consider adding these keywords to your resume:")
                st.write(', '.join(missing_keywords))
            else:
                st.write("Great job! Your resume includes all the important keywords from the job description.")

            # Visualizations
            st.subheader('Visualizations')
            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                st.write("Word Cloud")
                st.pyplot(generate_word_cloud(resume_text + " " + job_description))

            with viz_col2:
                st.write("Skill Comparison")
                radar_chart = generate_skill_radar_chart(resume_keywords, job_keywords)
                if radar_chart:
                    st.pyplot(radar_chart)

            st.write("Keyword Frequency")
            st.pyplot(generate_keyword_frequency_chart(resume_text, job_description))

            # Recommendations
            st.subheader('Recommendations')
            if score_value < 5:
                st.write("To improve your ATS score:")
                st.write("1. Add the missing keywords to your resume where applicable.")
                st.write("2. Ensure your resume is tailored to the specific job description.")
                st.write("3. Use industry-standard terminology and phrases.")
                st.write("4. Quantify your achievements where possible.")
            else:
                st.write("Excellent work! Your resume is well-matched to this job description.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        with col2:
            st.write("Please upload your resume and enter the job description, then click 'Start Analysis' to begin.")

if __name__ == '__main__':
    main()
