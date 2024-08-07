import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from rag import get_page_content, get_chunks, get_embeddings, retrieve_embedding, semantic_search, get_text, generate_response_rag, generate_response, vector_store, UI

def test_get_page_content():
    """
    Test if get_page_content correctly combines instruction and output fields.
    """
    sample_data = {'instruction': 'Test instruction', 'output': 'Test output'}
    page_content = get_page_content(sample_data)
    expected_content = "instruction: Test instruction\noutput: Test output"
    assert page_content == expected_content

def test_get_chunks():
    """
    Test if get_chunks correctly splits the text into chunks.
    """
    sample_text = "Test instruction 1\noutput: Test output 1\n\nTest instruction 2\noutput: Test output 2"
    expected_chunks = [
        'Test instruction 1\noutput: Test output 1\nTest instruction 2\noutput: Test output 2'
    ]
    chunks = get_chunks(sample_text)
    assert chunks == expected_chunks

@patch('rag.OpenAI')
def test_get_embeddings(mock_openai):
    """
    Test if get_embeddings correctly generates embeddings for chunk data.
    """
    sample_chunks = [
        'Test instruction 1\noutput: Test output 1\nTest instruction 2\noutput: Test output 2'
    ]
    expected_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_openai().embeddings.create.return_value.data = [{'embedding': [0.1, 0.2, 0.3]}, {'embedding': [0.4, 0.5, 0.6]}]

    embeddings = get_embeddings(sample_chunks)
    assert embeddings == expected_vectors

@patch('rag.index')
def test_retrieve_embedding(mock_index):
    """
    Test if retrieve_embedding correctly retrieves embeddings from the vector database.
    """
    expected_data = {
        'instruction': [0, 1],
        'output': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'text': ['chunk 1', 'chunk 2']
    }
    mock_index.fetch.return_value = {
        'vectors': {
            'vec_0': {'metadata': {'text': 'chunk 1'}, 'values': [0.1, 0.2, 0.3]},
            'vec_1': {'metadata': {'text': 'chunk 2'}, 'values': [0.4, 0.5, 0.6]}
        }
    }
    df = retrieve_embedding(mock_index, 2)
    assert df['instruction'].tolist() == expected_data['instruction']
    assert df['output'].tolist() == expected_data['output']


def test_health_endpoint(client):
    """
    Test the health check endpoint to ensure it's working correctly.
    """
    response = client.get('/')
    assert response.status_code == 200
