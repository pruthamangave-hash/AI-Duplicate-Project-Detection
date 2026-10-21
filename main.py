import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# -----------------------------------------
# PAGE NAVIGATION
# -----------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

# -----------------------------------------
# HOME PAGE
# -----------------------------------------

if st.session_state.page == "home":

    st.markdown("<br><br>", unsafe_allow_html=True)

    # University Logo
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(
            "data/assets/vu_logo.png",
            use_container_width=True
        )

    # Project Title
    st.markdown(
        """
        <h1 style="text-align:center;">
            AI-Based Duplicate Project Detection System
        </h1>

        <p style="text-align:center; font-size:20px;">
            Semantic Similarity Analysis for Academic Projects
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Start Button
    left, middle, right = st.columns([2, 1, 2])

    with middle:
        if st.button("START", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.stop()

# -----------------------------------------
# RETURN TO HOME BUTTON
# -----------------------------------------

if st.button("← Home"):
    st.session_state.page = "home"
    st.rerun()

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Duplicate Project Detection",
    page_icon="🔍",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🔍 AI-Based Duplicate Project Detection System")

st.write(
    "Check a new academic project against existing projects "
    "using semantic similarity."
)


# -----------------------------
# Load Dataset
# -----------------------------

projects = pd.read_csv("data/projects.csv")

projects["combined_text"] = (
    projects["Project Title"].fillna("") + ". " +
    projects["Abstract"].fillna("") + ". " +
    projects["Problem Statement"].fillna("") + ". " +
    projects["Objectives"].fillna("") + ". " +
    projects["Methodology"].fillna("") + ". " +
    projects["Technologies Used"].fillna("") + ". " +
    projects["Expected Outcome"].fillna("")
)


# -----------------------------
# Load SBERT
# -----------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# -----------------------------
# Existing Project Embeddings
# -----------------------------

@st.cache_data
def generate_embeddings(texts):
    return model.encode(texts)


embeddings = generate_embeddings(
    projects["combined_text"].tolist()
)


# -----------------------------
# Similarity Classification
# -----------------------------

def classify_similarity(score):

    if score >= 85:
        return "Potential Duplicate"

    elif score >= 70:
        return "High Similarity"

    elif score >= 40:
        return "Moderate Similarity"

    else:
        return "Low Similarity"


# -----------------------------
# PDF Extraction Function
# -----------------------------

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Input Mode
# -----------------------------

st.header("Choose Input Method")

input_mode = st.radio(
    "How do you want to submit the new project?",
    [
        "Enter Project Details Manually",
        "Upload Project Document"
    ]
)


new_project = ""


# -----------------------------
# Manual Input
# -----------------------------

if input_mode == "Enter Project Details Manually":

    st.subheader("Enter New Project Details")

    title = st.text_input("Project Title")

    abstract = st.text_area("Abstract")

    problem = st.text_area("Problem Statement")

    objectives = st.text_area("Objectives")

    methodology = st.text_area("Methodology")

    technologies = st.text_input("Technologies Used")

    outcome = st.text_area("Expected Outcome")

    new_project = (
        title + ". " +
        abstract + ". " +
        problem + ". " +
        objectives + ". " +
        methodology + ". " +
        technologies + ". " +
        outcome
    )


# -----------------------------
# Document Upload
# -----------------------------

else:

    st.subheader("Upload Project Document")

    uploaded_file = st.file_uploader(
        "Upload a TXT or PDF project document",
        type=["txt", "pdf"]
    )

    if uploaded_file is not None:

        if uploaded_file.type == "text/plain":

            new_project = uploaded_file.read().decode("utf-8")

        elif uploaded_file.type == "application/pdf":

            new_project = extract_pdf_text(uploaded_file)

        if new_project.strip():

            st.success("Project document text extracted successfully.")

            with st.expander("View Extracted Text"):

                st.write(new_project)

        else:

            st.warning(
                "No readable text could be extracted from this document."
            )


# -----------------------------
# Similarity Button
# -----------------------------

if st.button("🔍 CHECK SIMILARITY"):

    if new_project.strip() == "":

        st.warning(
            "Please enter project details or upload a project document."
        )

    else:

        new_embedding = model.encode(
            [new_project]
        )

        similarities = cosine_similarity(
            new_embedding,
            embeddings
        )[0]

        results = pd.DataFrame({

            "Project Title":
                projects["Project Title"],

            "Similarity":
                similarities * 100

        })

        results = results.sort_values(
            by="Similarity",
            ascending=False
        )

        results["Category"] = results[
            "Similarity"
        ].apply(classify_similarity)

        results["Similarity"] = results[
            "Similarity"
        ].round(2)


        # -----------------------------
        # Display Results
        # -----------------------------

        st.header("Similarity Results")

        top_project = results.iloc[0]

        st.subheader("Highest Similar Project")

        st.write(
            "**Project:**",
            top_project["Project Title"]
        )

        st.write(
            "**Similarity:**",
            str(top_project["Similarity"]) + "%"
        )

        st.write(
            "**Category:**",
            top_project["Category"]
        )


        st.subheader("Ranked Project Results")

        results.insert(
            0,
            "Rank",
            range(1, len(results) + 1)
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )