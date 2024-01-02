import os.path

import pandas as pd
import gdown
from langchain_community.vectorstores.chroma import Chroma
from tqdm import tqdm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


if not os.path.isfile("SPOTIFY_REVIEWS.csv"):
    gdown.download("https://drive.google.com/uc?id=1_xaRB6d2K_9-1dUmdU0GjtaqPO7uQnTM")

df = pd.read_csv("SPOTIFY_REVIEWS.csv")
df.drop(columns=['Unnamed: 0', 'review_id', 'pseudo_author_id', 'author_name'])
df.to_csv("SPOTIFY_REVIEWS_CLEANED.csv")
print("loading data...")
loader = CSVLoader(
    file_path="SPOTIFY_REVIEWS_CLEANED.csv",
    )

data = loader.load()

print("data loaded")

# split the loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

docs = text_splitter.split_documents(data)

# create the vector db to store all the split chunks as embeddings
embeddings = HuggingFaceEmbeddings()

data_size = len(data)
num_processes = 101
current_start = 0
current_end = current_start + data_size // num_processes

for i in tqdm(range(num_processes), desc="Processing chunks"):
    if current_end > data_size:
        current_end = data_size - 1
    all_splits = text_splitter.split_documents(data[current_start:current_end])

    # Update the progress bar description
    tqdm.write(f"Processing chunk {i + 1}/{num_processes}")

    current_start = current_end
    current_end = current_start + data_size // num_processes

    # Create a Milvus connection and store embeddings
    vector_db = Chroma.from_documents(
        all_splits,
        embeddings,
        persist_directory="chroma_db",
    )
