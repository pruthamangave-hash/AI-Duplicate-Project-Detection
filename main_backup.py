import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
    "Enter the details of a new project to find similar "
    "projects from the existing project dataset."
)


# -----------------------------
# Load Dataset
# -----------------------------

projects = pd.read_csv("data/projects.csv")


# Combine project fields

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
# Load Sentence-BERT Model
# -----------------------------

@st.cache_resource
def load_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


model = load_model()


# -----------------------------
# Generate Existing Embeddings
# -----------------------------

@st.cache_data
def generate_embeddings(texts):

    embeddings = model.encode(texts)

    return embeddings


embeddings = generate_embeddings(
    projects["combined_text"].tolist()
)


# -----------------------------
# User Input
# -----------------------------

st.header("Enter New Project Details")


title = st.text_input(
    "Project Title"
)


abstract = st.text_area(
    "Abstract"
)


problem = st.text_area(
    "Problem Statement"
)


objectives = st.text_area(
    "Objectives"
)


methodology = st.text_area(
    "Methodology"
)


technologies = st.text_input(
    "Technologies Used"
)


outcome = st.text_area(
    "Expected Outcome"
)


# -----------------------------
# Similarity Function
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
# Check Similarity Button
# -----------------------------

if st.button("🔍 CHECK SIMILARITY"):

    if title.strip() == "":

        st.warning("Please enter at least the Project Title.")

    else:

        new_project = (
            title + ". " +
            abstract + ". " +
            problem + ". " +
            objectives + ". " +
            methodology + ". " +
            technologies + ". " +
            outcome
        )


        # Generate embedding for new project

        new_embedding = model.encode(
            [new_project]
        )


        # Calculate similarity

        similarities = cosine_similarity(
            new_embedding,
            embeddings
        )[0]


        # Create result table

        results = pd.DataFrame({

            "Project Title":
                projects["Project Title"],

            "Similarity":
                similarities * 100

        })


        # Sort highest similarity first

        results = results.sort_values(
            by="Similarity",
            ascending=False
        )


        # Add classification

        results["Category"] = results[
            "Similarity"
        ].apply(classify_similarity)


        # Round percentage

        results["Similarity"] = results[
            "Similarity"
        ].round(2)


        # -----------------------------
        # Display Results
        # -----------------------------

        st.header("Similarity Results")


        # Highest match

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


        # Full ranking

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