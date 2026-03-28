import os.path

import pandas as pd
import gdown
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Maximum characters for a single review document before it is split.
# Play Store reviews are almost always well under this limit, so in practice
# every review remains one atomic chunk — preserving its full semantic context.
# Only unusually long reviews (e.g. copy-pasted essays) get split.
MAX_REVIEW_CHARS = 1500

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
print(f"Loaded {len(data)} reviews")

# Each document in `data` is already one complete review (one CSV row).
# We embed each review as a single chunk so retrieval returns whole, coherent
# reviews rather than fragments that may lose sentiment or context mid-sentence.
# The splitter is applied only as a safety net for outlier reviews that exceed
# MAX_REVIEW_CHARS; overlap is irrelevant for those rare cases but set
# conservatively to avoid mid-word cuts.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_REVIEW_CHARS,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " ", ""],  # prefer sentence boundaries
)
docs = text_splitter.split_documents(data)
split_count = len(docs) - len(data)
print(f"{len(docs)} chunks total ({split_count} extra from oversized reviews)")

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
