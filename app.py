import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined user credentials for login
USER_CREDENTIALS = {
    "user1": "password1",
    "user2": "password2"
}

# Example dataset with job details
def get_company_data():
    return pd.DataFrame({
        'Company': ['TechCorp', 'DataSol', 'FinServe', 'BioHealth'],
        'Industry': ['Technology', 'Technology', 'Finance', 'Healthcare'],
        'Location': ['Remote', 'On-site', 'Remote', 'Hybrid'],
        'Experience Level': ['Entry-level', 'Mid-level', 'Senior-level', 'Entry-level'],
        'Job Description': [
            'Software engineer with Python and machine learning skills.',
            'Data scientist with Python, SQL, and visualization expertise.',
            'Financial analyst with accounting and risk management experience.',
            'Biomedical engineer with healthcare technology experience.'
        ]
    })

# Function to extract text from uploaded PDF resume
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to calculate similarity between resume and job descriptions
def get_job_recommendations(resume_text, company_data):
    vectorizer = TfidfVectorizer()
    texts = [resume_text] + company_data['Job Description'].tolist()
    tfidf_matrix = vectorizer.fit_transform(texts)
    resume_vector = tfidf_matrix[0]
    job_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(resume_vector, job_vectors).flatten()
    company_data['Similarity Score'] = similarities
    return company_data.sort_values(by='Similarity Score', ascending=False)

# Main Streamlit app
def main():
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "application_success" not in st.session_state:
        st.session_state.application_success = False
    if "applied_companies" not in st.session_state:
        st.session_state.applied_companies = []

    # Login Page with background color
    if not st.session_state.authenticated:
        st.markdown("""
        <style>
            .login-page {
                background-color: #E8F4F8;
                padding: 40px;
                border-radius: 10px;
                border: 2px solid #A7C8D7;
            }
            .login-header {
                text-align: center;
                color: #2C6D8A;
            }
        </style>
        """, unsafe_allow_html=True)

        st.title("Login Page", anchor="login-header")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.success("Login successful! Welcome to the Job Recommendation System.")
            else:
                st.error("Invalid username or password.")
        return

    # Welcome Page with background color
    if "welcome_done" not in st.session_state or not st.session_state.welcome_done:
        st.markdown("""
        <style>
            .welcome-page {
                background-color: #F1F8FB;
                padding: 30px;
                border-radius: 10px;
            }
        </style>
        """, unsafe_allow_html=True)

        st.title("Welcome!")
        st.write("Welcome to the *Job Recommendation System*. Upload your resume, and we'll recommend jobs tailored to your profile.")
        if st.button("Get Started"):
            st.session_state.welcome_done = True
        return

    # Resume Upload Section with background color
    st.markdown("""
    <style>
        .upload-section {
            background-color: #F9F9F9;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=['pdf'])

    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text.strip() == "":
            st.error("Unable to extract text from the uploaded PDF. Please try a different file.")
            return
        
        st.subheader("Extracted Text from Your Resume")
        st.write(resume_text)

        company_data = get_company_data()
        recommendations = get_job_recommendations(resume_text, company_data)

        st.subheader("Recommended Jobs Based on Your Resume")
        for index, row in recommendations.iterrows():
            st.write(f"*{row['Company']}* ({row['Industry']}, {row['Location']}, {row['Experience Level']})")
            st.write(row['Job Description'])
            st.write(f"*Match Score:* {row['Similarity Score']:.2f}")
            if st.button(f"Apply to {row['Company']}", key=f"apply_{index}"):
                st.session_state.application_success = True
                st.session_state.applied_companies.append(row['Company'])
                st.success(f"Application to {row['Company']} submitted!")

    if st.session_state.application_success:
        st.markdown("""
        <style>
            .application-form {
                background-color: #D1ECF1;
                padding: 30px;
                border-radius: 10px;
                border: 2px solid #31708F;
            }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("Submit Your Details")
        with st.form("application_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            phone = st.text_input("Phone Number")
            feedback = st.text_area("Any Additional Feedback?")
            submitted = st.form_submit_button("Submit Application")

            if submitted:
                if not name or not email or not phone:
                    st.error("Please fill in all required fields.")
                elif "@" not in email or "." not in email:
                    st.error("Please enter a valid email address.")
                elif not phone.isdigit() or len(phone) < 8 or len(phone) > 15:
                    st.error("Phone number should contain only digits and be 8-15 characters long.")
                else:
                    st.balloons()
                    st.success("Your application has been successfully submitted!")
                    st.info("You will receive a response within 5 business days.")
                    
                    # Enhanced Thank-You Section
                    st.markdown("""
                    <div style="
                        background-color: #D1ECF1;
                        padding: 20px;
                        border-radius: 10px;
                        border: 2px solid #31708F;
                        text-align: center;">
                        <h2 style="color: #31708F; font-size: 24px;">🎉 Thank You for Applying! 🎉</h2>
                        <p style="color: #31708F; font-size: 18px;">We appreciate your application. Below are your details and the applied companies.</p>
                        <p style="color: #31708F; font-size: 16px;"><strong>You will receive a response within 5 business days.</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("Your Application Details")
                    st.write(f"*Name:* {name}")
                    st.write(f"*Email:* {email}")
                    st.write(f"*Phone:* {phone}")
                    st.write(f"*Feedback:* {feedback or 'No feedback provided.'}")

                    st.subheader("Applied Companies")
                    for company in st.session_state.applied_companies:
                        st.write(f"- {company}")

if _name_ == "_main_":
    main()
