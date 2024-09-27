import streamlit as st
from chain import Chain

st.set_page_config(page_title="StoryWeaver", layout="wide")

st.title("📖StoryWeaver: Novel Chapter Generator")
st.write("Generate a novel chapter based on your prompt. Results will vary based on the specificity of the prompt")

with st.form(key='input_form'):
    user_prompt = st.text_area(
        "Enter your prompt",
        height=150,
        placeholder="e.g., Generate an urban fantasy chapter 1 introducing Elmville and its residents James and Amber."
    )
    submit_button = st.form_submit_button(label='Generate Chapter')

if submit_button:
    if not user_prompt:
        st.error("Please enter a prompt.")
    else:
        chain = Chain()
        with st.spinner("Processing your prompt..."):
            try:
                output, genre, characters_str, places_str = chain.generate_chapter(user_prompt)
                st.subheader("Generated Chapter")
                st.write(output)

                # Optionally display extracted details
                with st.expander("Extracted Details"):
                    st.write(f"**Genre**: {genre}")
                    st.write(f"**Characters**: {characters_str}")
                    st.write(f"**Places**: {places_str}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
