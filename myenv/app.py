import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from matplotlib.backends.backend_agg import RendererAgg
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline
import sqlite3

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Initialize the question-answering pipeline with the specified model
qa_pipeline = pipeline("question-answering", model=model_name)

# Function to convert image to base64
def image_to_base64(image):
    with open(image, 'rb') as file:
        encoded_image = base64.b64encode(file.read()).decode()
    return encoded_image

logo_image = "TUP_LOGO.png"
image = Image.open(logo_image)

# Resize the image to the desired smaller size
resized_image = image.resize((100, 100))

# Convert the resized image to base64
encoded_image = image_to_base64(logo_image)

_lock = RendererAgg.lock

# Function to check login credentials
def authenticate(username, password):
    # Add your authentication logic here
    if username == "admin" and password == "password":
        return True
    else:
        return False

# Function to perform question answering with context segmentation
def answer_question(context, question):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    # Tokenize the question and context
    encoded_inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # Segment the input into chunks of size 512 (excluding special tokens)
    chunk_size = 512 - tokenizer.num_special_tokens_to_add(pair=True)
    chunks = [input_ids[:, i:i+chunk_size] for i in range(0, input_ids.size(1), chunk_size)]
    chunk_masks = [attention_mask[:, i:i+chunk_size] for i in range(0, attention_mask.size(1), chunk_size)]

    answers = []
    for chunk, mask in zip(chunks, chunk_masks):
        with torch.no_grad():
            outputs = model(input_ids=chunk, attention_mask=mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        # Find the start and end indices with the maximum scores
        start_index = torch.argmax(start_scores, dim=1).item()
        end_index = torch.argmax(end_scores, dim=1).item()

        # Convert token indices to tokens and concatenate the answer
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(chunk[0, start_index:end_index+1]))
        answers.append(answer)

    # Concatenate the answers from all chunks
    answer = " ".join(answers)
    return answer


# Function to read and display Excel data with question-answering feature
def display_excel_data(file_path, conn):
    df = pd.read_excel(file_path)
    st.dataframe(df)

    # Get column names from the first row of the DataFrame
    categories = df.columns.tolist()
    col1, col2 = st.columns(2)

    if not st.session_state["logged_in"]:
        with col1:
            st.markdown(
                f"""
                <style>
                .login-container {{
                    max-width: 500px;
                    margin: 0 auto;
                }}
                </style>

                <div class="login-container">
                    <h1 style='text-align: center;'>
                        TUP | Data Filter
                    </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

    
    with col1:
        selected_category = st.selectbox("Select a category", categories)
        if st.button("Apply Dropdown Filter"):
            with _lock:
                filtered_df = df[[selected_category]]
                filtered_df = filtered_df.dropna()
                st.subheader("Filtered Data (Dropdown Filter)")
                st.dataframe(filtered_df)

                # Create a simple bar graph for demonstration purposes
                st.subheader("Filtered Data Graph (Dropdown Filter)")
                sns.set_theme()
                fig, ax = plt.subplots()
                ax = sns.countplot(data=filtered_df, x=selected_category)
                ax.set_xlabel(selected_category)
                ax.set_ylabel("Count")
                ax.set_title("Filtered Data Count Plot")
                st.pyplot(fig)

    with col2:
        search_input = st.text_input("Search Filter")
        if st.button("Apply Search Filter"):
            with _lock:
                if search_input:
                    filtered_df = df.apply(lambda row: row.astype(str).str.contains(search_input, case=False).any(), axis=1)
                    filtered_df = df[filtered_df]
                    if filtered_df.empty:
                        st.write("No results found.")
                    else:
                        st.subheader("Filtered Data (Search Filter)")

                        # Apply highlighting to search results
                        def highlight_search_results(value):
                            if search_input.lower() in str(value).lower():
                                return "background-color: yellow"
                            else:
                                return ""

                        styled_df = filtered_df.style.applymap(highlight_search_results)
                        st.dataframe(styled_df)

    # Question-answering feature
    question = st.text_input("Ask a question about the data:")
    if st.button("Get Answer"):
        with _lock:
            if question:
                # Prepare the context for question answering
                context = "\n".join([f"{index+1}. " + " | ".join(row.astype(str)) for index, row in df.iterrows()])
                answer = answer_question(context, question)
                st.write("Answer:", answer)

# Function to create the file history table in the database
def create_file_history_table(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS file_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

# Function to store the uploaded file name in the file history table
def store_file_history(conn, file_name):
    cursor = conn.cursor()
    # Check if the file name already exists in the table
    cursor.execute(
        """
        SELECT * FROM file_history WHERE file_name = ?
        """,
        (file_name,),
    )
    existing_file = cursor.fetchone()
    if existing_file is None:
        cursor.execute(
            """
            INSERT INTO file_history (file_name) VALUES (?)
            """,
            (file_name,),
        )
        conn.commit()

# Function to retrieve the file history from the database
def retrieve_file_history(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT * FROM file_history ORDER BY timestamp DESC
        """
    )
    return cursor.fetchall()

# Function to get the selected file path from the file history sidebar
def get_selected_file_path(file_history):
    selected_file = st.sidebar.selectbox("Select a file from history", ["None"] + [file[1] for file in file_history])
    file_path = None
    if selected_file != "None":
        file_path = selected_file
    return file_path

# Main function
def main():
    # Check if user is already logged in
    if "username" not in st.session_state:
        st.session_state["username"] = ""
        st.session_state["logged_in"] = False

    # Login page
    if not st.session_state["logged_in"]:
        st.set_page_config(
            page_title='TUP | Data Filter',
            page_icon=image,
            layout="centered"
        )

        st.markdown(
            f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{encoded_image}" alt="image" width="150" height="150"></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <h1 style='text-align:center;'>
                TUP | Data Filter
            </h1>
            <h2 style='text-align: center;'>Login</h2>
            """,
            unsafe_allow_html=True
        )

        # Add styling to the login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")

            if login_btn:
                if authenticate(username, password):
                    st.session_state["username"] = username
                    st.session_state["logged_in"] = True
                    st.success("Logged in successfully, Please click the button again to proceed!")
                else:
                    st.error("Invalid username or password, Please try again.")

        return


    st.set_page_config(
        page_title='TUP | Data Filter',
        page_icon=image,
        layout="wide"
    )

    # Custom CSS for the sidebar
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            position: absolute;
            top: 0;
            right: 0;
            z-index: 999;
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.markdown(f"Logged in as: **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False

    # Database connection
    conn = sqlite3.connect("file_history.db")
    create_file_history_table(conn)

    # Retrieve file history from the database
    file_history = retrieve_file_history(conn)

    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    # Get the selected file path from the file history sidebar
    selected_file_path = get_selected_file_path(file_history)

    if uploaded_file is not None and selected_file_path is not None:
        st.warning("Please upload a file or select a file from the history sidebar, but not both.")
    elif uploaded_file is not None:
        # Check if the file is already in the file history
        existing_file = next((file for file in file_history if file[1] == uploaded_file.name), None)
        if existing_file is None:
            # Store the uploaded file in the file history
            store_file_history(conn, uploaded_file.name)
        # Display uploaded file and apply filtering
        display_excel_data(uploaded_file.name, conn)
    elif selected_file_path is not None:
        # Display the selected file and apply filtering
        display_excel_data(selected_file_path, conn)

    # Close the database connection
    conn.close()

# Run the app
if __name__ == "__main__":
    main()
