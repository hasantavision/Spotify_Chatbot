import os.path

import pandas as pd
import gdown
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

CHROMA_PERSIST_DIR = "data/chroma_db"
REVIEWS_RAW = "SPOTIFY_REVIEWS.csv"
REVIEWS_CLEANED = "SPOTIFY_REVIEWS_CLEANED.csv"

if not os.path.isfile(REVIEWS_RAW):
    gdown.download("https://drive.google.com/uc?id=1_xaRB6d2K_9-1dUmdU0GjtaqPO7uQnTM")

df = pd.read_csv(REVIEWS_RAW)
# Fix: reassign after drop so columns are actually removed
df = df.drop(columns=["Unnamed: 0", "review_id", "pseudo_author_id", "author_name"], errors="ignore")
df.to_csv(REVIEWS_CLEANED, index=False)

print("Loading data...")
loader = CSVLoader(file_path=REVIEWS_CLEANED)
data = loader.load()
print(f"Loaded {len(data)} documents")

# Best practice: smaller chunks (512 tokens) with meaningful overlap (20%)
# improves retrieval precision — large chunks hurt both recall and context quality
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
docs = text_splitter.split_documents(data)
print(f"Split into {len(docs)} chunks")

# BAAI/bge-base-en-v1.5 consistently ranks at the top of MTEB benchmarks;
# normalize_embeddings=True enables cosine similarity comparisons
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

# Batch insertion to manage memory; process chunks of `docs` (already split),
# not raw `data` re-split in every iteration
BATCH_SIZE = 500
num_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(num_batches), desc="Indexing batches"):
    batch = docs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    tqdm.write(f"Batch {i + 1}/{num_batches}: {len(batch)} chunks")

    if i == 0:
        # Create the DB on the first batch
        vector_db = Chroma.from_documents(
            batch,
            embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    else:
        # Add to the existing DB for all subsequent batches
        vector_db.add_documents(batch)

print(f"Indexing complete. Vector DB persisted at '{CHROMA_PERSIST_DIR}'")
