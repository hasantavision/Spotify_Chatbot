import os.path

import pandas as pd
import gdown
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Maximum characters for a single review before it is split.
# Play Store reviews are almost always well under this — so every review
# stays as one atomic chunk preserving full sentiment context.
MAX_REVIEW_CHARS = 1500

CHROMA_PERSIST_DIR = "data/chroma_db"
REVIEWS_RAW = "SPOTIFY_REVIEWS.csv"
REVIEWS_CLEANED = "SPOTIFY_REVIEWS_CLEANED.csv"

if not os.path.isfile(REVIEWS_RAW):
    gdown.download("https://drive.google.com/uc?id=1_xaRB6d2K_9-1dUmdU0GjtaqPO7uQnTM")

df = pd.read_csv(REVIEWS_RAW)
df = df.drop(columns=["Unnamed: 0", "review_id", "pseudo_author_id", "author_name"], errors="ignore")
df.to_csv(REVIEWS_CLEANED, index=False)

print(f"Loaded {len(df)} reviews — building Documents with structured metadata…")


def _first_valid(row, *candidates):
    """Return the first non-null value from a list of candidate column names."""
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return None


docs_raw: list[Document] = []
for _, row in df.iterrows():
    text = _first_valid(row, "content", "review_text", "text", "review")
    if not text:
        continue

    # Store structured fields as Chroma metadata so SelfQueryRetriever can
    # apply natural-language filters: "1-star reviews", "issues in 2023", etc.
    metadata: dict = {}
    rating = _first_valid(row, "score", "rating", "stars")
    if rating is not None:
        try:
            metadata["rating"] = int(rating)
        except (ValueError, TypeError):
            pass
    date_raw = _first_valid(row, "at", "date", "review_date", "created_at", "reviewedAt")
    if date_raw is not None:
        metadata["date"] = str(date_raw)[:10]  # YYYY-MM-DD
    version = _first_valid(row, "reviewCreatedVersion", "app_version", "version", "appVersion")
    if version is not None:
        metadata["app_version"] = str(version)

    docs_raw.append(Document(page_content=str(text).strip(), metadata=metadata))

print(f"Built {len(docs_raw)} review documents")

# Safety-net splitter only fires for reviews that exceed MAX_REVIEW_CHARS.
# chunk_overlap=0 because there is no benefit in duplicating text within
# a single review; sentence-boundary separators avoid mid-word cuts.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_REVIEW_CHARS,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " ", ""],
)
docs = text_splitter.split_documents(docs_raw)
split_count = len(docs) - len(docs_raw)
print(f"{len(docs)} total chunks ({split_count} extra from oversized reviews)")

# BAAI/bge-base-en-v1.5: top MTEB benchmark; normalize for cosine similarity
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

BATCH_SIZE = 500
num_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(num_batches), desc="Indexing batches"):
    batch = docs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    tqdm.write(f"Batch {i + 1}/{num_batches}: {len(batch)} chunks")
    if i == 0:
        vector_db = Chroma.from_documents(
            batch, embeddings, persist_directory=CHROMA_PERSIST_DIR
        )
    else:
        vector_db.add_documents(batch)

print(f"Indexing complete. Vector DB persisted at '{CHROMA_PERSIST_DIR}'")
