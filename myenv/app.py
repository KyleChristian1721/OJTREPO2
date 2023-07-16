import streamlit as st
import matplotlib
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from matplotlib.backends.backend_agg import RendererAgg
import sqlite3
import os
import numpy as np

# Function to convert image to base64
def image_to_base64(image):
    with open(image, 'rb') as file:
        encoded_image = base64.b64encode(file.read()).decode()
    return encoded_image

logo_image = "https://github.com/KyleChristian1721/OJTREPO2/blob/master/myenv/TUP_LOGO.png"
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


def convert_input_to_numeric(input_value):
    try:
        return int(input_value)
    except ValueError:
        try:
            return float(input_value)
        except ValueError:
            return np.nan
        
# Function to read and display Excel data with question-answering feature

def display_excel_data(file_path, conn):
    chunk_size = 10000  # Adjust the chunk size based on your system's capabilities

    with pd.ExcelFile(file_path) as xls:
        sheet_names = xls.sheet_names
        sheet_name = st.selectbox("", sheet_names)
        df = xls.parse(sheet_name, engine="openpyxl")

        # Iterate over the rows in chunks
        for i in range(0, len(df), chunk_size):
            df_chunk = df.iloc[i : i + chunk_size]
            st.dataframe(df_chunk)

        # Get column names from the first row of the DataFrame
        categories = df_chunk.columns.tolist()
        col2 = st.columns(1)

        if not st.session_state["logged_in"]:
            col2[0].markdown(
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

        with col2[0]:
            categories = st.multiselect("Select categories for filtering", ["Global"] + categories)
            search_inputs = []
            for category in categories:
                search_inputs.append(st.text_input(f"Search Filter ({category})", key=f"search_input_{category}"))

            selected_Barcategory = st.selectbox("Select category for bar graph", df_chunk.columns)

            if st.button("Apply Search Filter"):
                with _lock:
                    filtered_df = df_chunk.copy()
                    for category, search_input in zip(categories, search_inputs):
                        if search_input:
                            if category == "Global":
                                search_mask = filtered_df.apply(lambda row: any(
                                    str(search_input.strip()).lower() in str(value).strip().lower()
                                    for value in row
                                ), axis=1)
                            else:
                                if pd.api.types.is_numeric_dtype(filtered_df[category]):
                                    search_mask = filtered_df[category] == convert_input_to_numeric(search_input)
                                else:
                                    search_mask = filtered_df[category].astype(str).str.strip().str.lower() == str(search_input.strip()).lower()
                                filtered_df = filtered_df[search_mask]

                    if filtered_df.empty:
                        st.write("No results found.")
                    else:
                        st.subheader("Filtered Data (Search Filter)")

                        # Apply highlighting to search results
                        def highlight_search_results(value, category):
                            if category == "Global":
                                if any(
                                    str(search_input.strip()).lower() in str(value).strip().lower()
                                    for search_input in search_inputs
                                ):
                                    return "background-color: yellow"
                            else:
                                if any(
                                    str(search_input.strip()).lower() == str(value).strip().lower()
                                    and str(category) == str(value)
                                    for search_input, category in zip(search_inputs, categories)
                                ):
                                    return "background-color: yellow"
                            return ""

                        styled_df = filtered_df.style.applymap(lambda x: highlight_search_results(x, categories)).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'yellow')]}])
                        st.write(styled_df)

                    # Display bar graph based on filtered data
                    st.subheader("Filtered Data Bar Graph")

                    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
                    ax.bar(filtered_df[selected_Barcategory].value_counts().index, filtered_df[selected_Barcategory].value_counts())
                    ax.set_xlabel(selected_Barcategory)
                    ax.set_ylabel("Count")
                    ax.set_title("Filtered Data Bar Graph")
                    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels if needed
                    ax.spines["top"].set_visible(False)  # Remove top border
                    ax.spines["right"].set_visible(False)  # Remove right border
                    plt.tight_layout()  # Adjust spacing

                    st.pyplot(fig)

                    # Export filtered data
                    csv = filtered_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()

                    st.download_button("Download CSV", data=csv, file_name="filtered_data.csv")




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
        file_path = os.path.join("upload_folder", selected_file)
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
    st.set_option('deprecation.showPyplotGlobalUse', False)

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
        # Save the uploaded file to the desired location
        file_bytes = uploaded_file.read()
        with open(os.path.join("upload_folder", uploaded_file.name), "wb") as file:
            file.write(file_bytes)
        # Display uploaded file and apply filtering
        display_excel_data(os.path.join("upload_folder", uploaded_file.name), conn)
    elif selected_file_path is not None:
        # Display the selected file and apply filtering
        display_excel_data(selected_file_path, conn)

    # Close the database connection
    conn.close()

# Run the app
if __name__ == "__main__":
    main()