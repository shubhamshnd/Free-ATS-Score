# Free Resume Matching Tool

This is a simple tool built using Streamlit, Spacy, and Scikit-learn to match resumes with job descriptions. It calculates a matching score based on the similarity between the resume and the job description and provides an Applicant Tracking System (ATS) score to evaluate the match.

## How to Use

1. Clone the repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run `python -m spacy download en_core_web_sm`
4. Run the Streamlit app using `streamlit run app.py`.
5. Upload your resume file and enter the job description text.
6. The tool will extract keywords from the job description, calculate the matching score, and provide an ATS score based on the match rate.

## Requirements

```
pip install streamlit spacy scikit-learn PyPDF2 matplotlib seaborn wordcloud pandas numpy
python -m spacy download en_core_web_sm
```

## File Structure

- `main.py`: The main Streamlit application file.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file providing information about the project.

## Usage

Feel free to fork and modify this project according to your requirements. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
