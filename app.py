import streamlit as st
import pandas as pd
import io
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration ---
# Sheet names to be considered for RAG
RAG_SHEET_NAMES = ["QT Register 2025", "2025 INV", "Meeting Agenda", "Payment Pending"]
# OpenAI Embedding Model
EMBEDDING_MODEL = "text-embedding-ada-002"
# OpenAI LLM Model for responses
LLM_MODEL = "gpt-3.5-turbo"
# Number of top relevant chunks to retrieve for the LLM
TOP_K_CHUNKS = 3

# --- Helper Functions for RAG ---

@st.cache_data(show_spinner="Processing Excel data and generating embeddings (this may take a moment)...")
def load_and_process_excel_for_rag(uploaded_file, api_key):
    """
    Loads specified sheets from an Excel file, extracts relevant text,
    and generates embeddings for each text chunk.
    """
    if not uploaded_file:
        return None, None

    all_chunks = []
    all_embeddings = []

    try:
        # Set OpenAI API key for embedding generation within this function
        openai.api_key = api_key

        xls = pd.ExcelFile(uploaded_file)

        for sheet_name in RAG_SHEET_NAMES:
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)

                # Dynamically determine columns to combine for text chunk based on sheet
                if sheet_name == "QT Register 2025":
                    # Columns to consider: 'Quote No', 'Company Name', 'Product', 'Quote Value', 'Status', 'Sales Person', 'Remarks'
                    relevant_cols = ['Quote No', 'Company  Name', 'Product', 'Quote Value(BHD)', 'Status', 'Sales Person', 'Remarks']
                elif sheet_name == "2025 INV":
                    # Columns to consider: 'SSS Invoice No', 'To', 'Description', 'Grand Total in BHD', 'Payment terms'
                    relevant_cols = ['SSS Invoice No', 'To', 'Description', 'Grand Total in BHD', 'Payment terms']
                elif sheet_name == "Meeting Agenda":
                    # Columns to consider: 'Date', 'Topic', 'Attendees', 'Action Items'
                    # Note: Meeting Agenda has 2 header rows, so actual data starts from row 3 (index 2) if read without skiprows
                    # Assuming pandas reads it correctly when sheet_name is specified.
                    relevant_cols = ['Date', 'Topic', 'Attendees', 'Action Items']
                elif sheet_name == "Payment Pending":
                    # Columns to consider: 'SSS Invoice No', 'Company Name', 'Amount Pending', 'Due Date'
                    # Note: Payment Pending has 1 header row, so actual data starts from row 2 (index 1) if read without skiprows
                    relevant_cols = ['SSS Invoice No', 'Company Name', 'Amount Pending', 'Due Date']
                else:
                    relevant_cols = df.columns.tolist() # Fallback to all columns

                # Filter for existing columns to avoid KeyError
                existing_cols = [col for col in relevant_cols if col in df.columns]
                
                # Create text chunks from each row
                for index, row in df.iterrows():
                    chunk_parts = []
                    for col in existing_cols:
                        # Ensure all values are converted to string to avoid errors in f-string
                        chunk_parts.append(f"{col}: {str(row[col])}")
                    
                    text_chunk = f"From {sheet_name} - " + ", ".join(chunk_parts)
                    all_chunks.append({"content": text_chunk, "source": sheet_name})

                    # Generate embedding for the chunk
                    try:
                        embedding_response = openai.embeddings.create(
                            input=text_chunk,
                            model=EMBEDDING_MODEL
                        )
                        all_embeddings.append(embedding_response.data[0].embedding)
                    except openai.APIError as e:
                        st.error(f"OpenAI API Error during embedding generation: {e}")
                        return None, None
                    except Exception as e:
                        st.error(f"An unexpected error occurred during embedding generation: {e}")
                        return None, None
            else:
                st.warning(f"Sheet '{sheet_name}' not found in the uploaded Excel file. Skipping.")

    except pd.errors.ParserError as e:
        st.error(f"Error parsing Excel file: {e}. Please ensure it's a valid .xlsx file.")
        return None, None
    except FileNotFoundError: # This is handled by Streamlit's uploader if no file.
        st.error("Excel file not found. Please upload the file.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")
        return None, None
    
    if not all_chunks:
        st.warning("No data extracted from the specified sheets. Please check the Excel file content.")
        return None, None

    return all_chunks, np.array(all_embeddings)

def get_embedding(text, api_key):
    """Generates an embedding for the given text using OpenAI API."""
    try:
        openai.api_key = api_key # Ensure API key is set before call
        response = openai.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except openai.APIError as e:
        st.error(f"OpenAI API Error during query embedding: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during query embedding: {e}")
        return None

def retrieve_chunks(query_embedding, all_chunks, all_embeddings_np, top_k=TOP_K_CHUNKS):
    """
    Retrieves the top_k most similar chunks to the query embedding.
    """
    if query_embedding is None or all_embeddings_np is None or len(all_embeddings_np) == 0:
        return []

    # Calculate cosine similarity
    similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), all_embeddings_np)[0]
    
    # Get indices of top_k most similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1] # Get top_k and reverse to get highest first

    # Return the corresponding chunks
    retrieved = []
    for idx in top_indices:
        retrieved.append(all_chunks[idx])
    return retrieved

def ask_openai_llm(query, retrieved_chunks, api_key):
    """
    Asks the LLM a question, augmented with retrieved context.
    """
    if not api_key:
        st.error("OpenAI API Key is not provided.")
        return "Please enter your OpenAI API key to get responses."
    
    context = "\n".join([chunk["content"] for chunk in retrieved_chunks])
    
    if not context:
        return "No relevant context found in the Excel sheets for your query. Please try a different question or ensure the data contains relevant information."

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question ONLY based on the provided context from the Excel sheets. If the answer is not in the context, clearly state that you don't have enough information from the provided data."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    try:
        openai.api_key = api_key # Ensure API key is set before call
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=prompt_messages,
            temperature=0.1, # Low temperature for factual, less creative answers
            max_tokens=500 # Limit response length
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        st.error(f"OpenAI API Error during LLM completion: {e}")
        return "An error occurred while communicating with the OpenAI LLM. Please check your API key and network connection."
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM completion: {e}")
        return "An unexpected error occurred while processing your request."

# --- Streamlit UI ---
st.title("RAG Excel Data Query")
st.markdown("Upload your `SSS Master Sheet.xlsx` and ask questions about the data in specific sheets.")

# 1. Upload Excel File
uploaded_file = st.file_uploader("Upload your SSS Master Sheet.xlsx", type=["xlsx"])

# 2. Input OpenAI API Key
# Prioritize Streamlit secrets for deployment, fallback to text input for local
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("Enter your OpenAI API Key", type="password")

if "OPENAI_API_KEY" in st.secrets:
    st.success("OpenAI API Key loaded from Streamlit secrets.")
elif openai_api_key: # Only show warning if user hasn't entered it and it's not in secrets
    st.warning("No OpenAI API Key found in Streamlit secrets. Using provided key.")


# Store processed data in session state to avoid reprocessing on every rerun
if 'all_chunks' not in st.session_state:
    st.session_state.all_chunks = None
if 'all_embeddings_np' not in st.session_state:
    st.session_state.all_embeddings_np = None
if 'last_uploaded_file_id' not in st.session_state:
    st.session_state.last_uploaded_file_id = None
if 'last_api_key_for_processing' not in st.session_state:
    st.session_state.last_api_key_for_processing = None


# Process data when file is uploaded and API key is available
if uploaded_file and openai_api_key:
    # Use a flag to track if processing is needed
    needs_processing = False

    # Check if the file itself has changed
    if st.session_state.last_uploaded_file_id != (uploaded_file.id if uploaded_file else None):
        needs_processing = True
    
    # Check if the API key has changed
    if st.session_state.last_api_key_for_processing != openai_api_key:
        needs_processing = True

    if needs_processing:
        with st.spinner("Loading and processing data for RAG..."):
            st.session_state.all_chunks, st.session_state.all_embeddings_np = \
                load_and_process_excel_for_rag(uploaded_file, openai_api_key)
            
            if st.session_state.all_chunks is not None and st.session_state.all_embeddings_np is not None:
                st.success(f"Successfully processed {len(st.session_state.all_chunks)} data chunks from Excel.")
                st.session_state.last_uploaded_file_id = uploaded_file.id if uploaded_file else None
                st.session_state.last_api_key_for_processing = openai_api_key
            else:
                st.error("Failed to process Excel data. Please check the file and API key.")
                st.session_state.last_uploaded_file_id = None # Reset to force re-upload/re-entry
                st.session_state.last_api_key_for_processing = None

# 3. User Query Section
st.subheader("Ask a Question")
user_query = st.text_area("Type your question here:", "What is the status of quote SSS-QT-2025-001 from ABC Company?")

if st.button("Get Answer"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
    elif not uploaded_file:
        st.warning("Please upload the Excel file.")
    elif st.session_state.all_chunks is None or st.session_state.all_embeddings_np is None:
        st.warning("Data is still loading or failed to load. Please wait for processing to complete or re-upload.")
    elif not user_query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving information and generating response..."):
            # 1. Get query embedding
            query_embed = get_embedding(user_query, openai_api_key)

            if query_embed is not None:
                # 2. Retrieve relevant chunks
                retrieved_chunks = retrieve_chunks(
                    query_embed, 
                    st.session_state.all_chunks, 
                    st.session_state.all_embeddings_np, 
                    top_k=TOP_K_CHUNKS
                )

                # 3. Ask LLM with context
                response = ask_openai_llm(user_query, retrieved_chunks, openai_api_key)
                st.write("---")
                st.subheader("Answer:")
                st.write(response)

                if retrieved_chunks:
                    st.subheader("Relevant Sources:")
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**{i+1}. From Sheet:** `{chunk['source']}`")
                        with st.expander(f"**Chunk Content**"):
                            st.code(chunk['content'])
