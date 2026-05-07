from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
import streamlit as st

# Load API key
# load_dotenv()

# Streamlit page settings
st.set_page_config(page_title="College Assistant Chatbot")

# Title
st.title("🤖 AI College Assistant")

st.write("Ask anything below 👇")

# Groq model
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama3-8b-8192"
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful college assistant chatbot.

    Answer the following question clearly and shortly.

    Question: {input}
    """
)

# Create chain
chain = prompt | llm

# Chat input
user_input = st.chat_input("Type your message here...")

# If user enters message
if user_input:

    # Show user message
    st.chat_message("user").write(user_input)

    # ==============================
    # Attendance Calculator Feature
    # ==============================

    if user_input.lower().startswith("attendance"):

        try:
            parts = user_input.split()

            current_attendance = float(parts[1])
            remaining_classes = int(parts[2])

            total_classes = 100

            attended_classes = (
                current_attendance / 100
            ) * total_classes

            required_attendance = 75

            needed_attendance = (
                required_attendance
                * (total_classes + remaining_classes)
                / 100
            )

            classes_needed = int(
                needed_attendance - attended_classes
            )

            if classes_needed <= 0:

                result = (
                    "✅ You are already above 75% attendance."
                )

            elif classes_needed > remaining_classes:

                result = (
                    "❌ Even if you attend all remaining classes, "
                    "75% attendance is not possible."
                )

            else:

                result = (
                    f"📚 You must attend at least "
                    f"{classes_needed} out of "
                    f"{remaining_classes} remaining classes "
                    f"to reach 75% attendance."
                )

            st.chat_message("assistant").write(result)

        except:

            st.chat_message("assistant").write(
                "⚠️ Use format:\n\nattendance 68 20"
            )

    # ==============================
    # Normal AI Chatbot
    # ==============================

    else:

        try:

            response = chain.invoke(
                {"input": user_input}
            )

            st.chat_message("assistant").write(
                response.content
            )

        except Exception as e:

            st.chat_message("assistant").write(
                "⚠️ Error generating response. Please try again."
            )