import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from matplotlib.backends.backend_agg import RendererAgg

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

# Function to read and display Excel data
def display_excel_data(file):
    df = pd.read_excel(file)
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
                    <h2 style='text-align: center;'>Login</h2>
                    <div style='display: flex; justify-content: center;'>
                        <img src="data:image/png;base64,{image_to_base64(resized_image)}" width="100">
                    </div>
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
            <h1 style='text-align: center;'>
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
                st.success("Logged in successfully!")
                st.session_state["username"] = username
                st.session_state["logged_in"] = True
            else:
                st.error("Invalid username or password.")

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

    # Display the image with adjusted size and centered
    st.markdown(
        f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{encoded_image}" alt="image" width="150" height="150"></div>',
        unsafe_allow_html=True
    )

    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Display uploaded file and apply filtering
        display_excel_data(uploaded_file)

# Run the app
if __name__ == "__main__":
    main()
