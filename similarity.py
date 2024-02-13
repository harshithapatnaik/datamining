# -------------------------------------------------------------------------
# AUTHOR: Harshitha Patnaik
# FILENAME: similarity.py
# SPECIFICATION: Finds the most similar docs based on their cosine similarity
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: roughly a day
# -----------------------------------------------------------------------*/
# Importing some Python libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Defining the documents
doc1 = "soccer is my favorite sport"
doc2 = "I like sports and my favorite one is soccer"
doc3 = "support soccer at the olympic games"
doc4 = "I do like soccer, my favorite sport in the olympic games"

# Use the following words as terms to create your document-term matrix
terms = ["soccer", "favorite", "sport", "like", "one", "support", "olympic", "games"]

# Define a function to create document-term matrix
def create_document_term_matrix(documents, terms):
    matrix = []
    for doc in documents:
        doc.split()
        stripSpecialChar = [term.strip(',') for term in doc.split()]
        vector = [stripSpecialChar.count(term) for term in terms]
        matrix.append(vector)
    return np.array(matrix)

# Create document-term matrix
doc_term_matrix = create_document_term_matrix([doc1, doc2, doc3, doc4], terms)

# Compare the pairwise cosine similarities and store the highest one
similarities = cosine_similarity(doc_term_matrix)
max_similarity = np.max(similarities[np.triu_indices(len(doc_term_matrix), k=1)])

# Find the indices of the most similar documents
most_similar_indices = np.where(similarities == max_similarity)

# Print the highest cosine similarity following the information below
print(f"The most similar documents are: doc{most_similar_indices[0][0]+1} and doc{most_similar_indices[1][0]+1} with cosine similarity = {max_similarity:.3f}")
