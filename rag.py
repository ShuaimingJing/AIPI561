from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from getpass import getpass
from pinecone import Pinecone
import os
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


def get_page_content(row):
    instruction = row['instruction']
    output = row['output']
    page_content = f"instruction: {instruction}\noutput: {output}"
    return page_content


def get_chunks(text):
    """
    Function to get the chunks of text from the raw text

    Args:
        text (str): The raw text from the PDF file

    Returns:
        chunks (list): The list of chunks of text
    """

    # Initialize the text splitter
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len # Use the length function to get the length of the text
    )

    # Get the chunks of text
    chunks = splitter.split_text(text)

    return chunks


def get_embeddings(chunk_data):
    """
    Get the embedding vectors for the chunk data

    Arg:
    - chunk_data: a list of chunk data

    Return:
    - Embedded vectors

    """
    
    client = OpenAI(api_key = 'your_api_key')

    response = client.embeddings.create(
        input=chunk_data,
        model="text-embedding-3-small"
        )

    vectors_list = [item.embedding for item in response.data]
    return vectors_list



# Store vectors in vector database
def vector_store(vectors_list):
    # Iterate over the vectors_list
    for i in range(len(vectors_list)):
        index.upsert(
            vectors=[
                {
                    'id': f'vec_{i}',
                    'values': vectors_list[i],
                    'metadata': {"text":chunks[i]}
                }
            ],
        )


def retrieve_embedding(index, num_embed):
    """
    Convert the information of vectors in the database into a panda dataframe
    
    Args:
    - index: Name of vector database(already set up)
    - num_embed: total number of vectors in the vector databse

    Return:
    - a dataframe which contains the embedded vectors and corresponding text
    """
    # Initialize a dictionary to store embedding data
    embedding_data = {"id":[], "values":[], "text":[]}
    
    # Fetch the embeddings 
    embedding = index.fetch([f'vec_{i}' for i in range(num_embed)])
    
    for i in range(num_embed):
        embedding_data["id"].append(i)
        idx = f"vec_{i}"
        embedding_data["text"].append(embedding['vectors'][idx]['metadata']['text'])
        embedding_data["values"].append(embedding['vectors'][idx]['values'])
        
    return pd.DataFrame(embedding_data)



def semantic_search(query_vector, db_embeddings):
    """
    Find the top three vectors which have the highest comsine similarity with the query vector

    Args:
    - query_vector: embedded vector of user query
    - db_embeddings: embedded vectors from vector database

    Return:
    - The indices of top three most similar vectors with the query vector
    """
    
    similarities = cosine_similarity(query_vector, db_embeddings)[0]
    # Get the indices of the top three similarity scores
    top_3_indices = np.argsort(similarities)[-3:][::-1]  # This sorts and then reverses to get top 3
    # Retrieve the top three most similar chunks and their similarity scores
    
    return top_3_indices



def get_text(embedding_data, top_3_indices):
    """
    Extracts text corresponding to the given top vectors from embedding data.

    Args:
    - embedding_data (DataFrame): DataFrame containing columns 'id', 'values', and 'text'.
    - top_vectors (list): List of indices for which corresponding text needs to be extracted.

    Returns:
    - combined_text (str): Combined text corresponding to the top vectors.
    """
   # Extract text from selected rows
    selected_texts = embedding_data.loc[top_3_indices, 'text'].tolist()

    # Combine the selected texts into a single string
    combined_text = ' '.join(selected_texts)

    return combined_text


def generate_response_rag(user_input,context):

    client = OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key='your_api_key'
    )

    completion = client.chat.completions.create(
        model='LLaMA_CPP',
        messages=[
            {"role":"system","content":"Hello, how can I help you?"},
            {"role": "assistant", "content": context},
            {"role":"user","content":user_input}
        ]
    )

    return completion.choices[0].message.content

def generate_response(user_input):

    client = OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key='your_api_key'
    )

    completion = client.chat.completions.create(
        model='LLaMA_CPP',
        messages=[
            {"role":"system","content":"Hello, how can I help you?"},
            {"role":"user","content":user_input}
        ]
    )

    return completion.choices[0].message.content



def UI():
    st.title("Veterinary LLM")

    # Create a form for user input
    with st.form("input_form"):
        user_input = st.text_area("Enter your message:")
        submitted = st.form_submit_button("Submit")
        query = user_input
        client = OpenAI(api_key = 'your_api_key')

        query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
        )
        query_vector = [item.embedding for item in query_embedding.data]
        embedding_data = retrieve_embedding(index,len(vectors_list))
        top_3_indices = semantic_search(query_vector, vectors_list)
        context = get_text(embedding_data, top_3_indices)

    # Check if the question is sumbitedd by users
    if submitted:
        st.text("Response with RAG:")
        st.write(generate_response_rag(user_input,context))
        st.text("Response without RAG:")
        st.write(generate_response(user_input))



if __name__ == "__main__":
    df = pd.read_csv('/Users/shuai/Downloads/translated_veterinary.csv')
    loader = CSVLoader('/Users/shuai/Downloads/translated_veterinary.csv')
    data = loader.load()
    all_page_contents = df.apply(get_page_content, axis=1).tolist()
    text = "\n".join(all_page_contents)
    chunks = get_chunks(text)
    vectors_list = get_embeddings(chunks)

    pc = Pinecone(api_key='your_api_key')
    index = pc.Index("561rag")
    vector_store(vectors_list)
    UI()