# RAG Chatbot Streamlit App with Spotify Data

[![RAG chatbot with OpenAI](https://github.com/hasantavision/Spotify_Chatbot/blob/master/assets/yt1.png)](https://www.youtube.com/watch?v=fL4f9hgiwPw "RAG chatbot with OpenAI")
[![RAG chatbot with Llama2](https://github.com/hasantavision/Spotify_Chatbot/blob/master/assets/yt2.png)](https://www.youtube.com/watch?v=D-mgXB96r1o "RAG chatbot with Llama2")

## How to use this repo

### Clone the repo
```bash
git clone https://github.com/hasantavision/Spotify_Chatbot.git
```

### Install requirements

Make sure you have python ~ 3.9 - 3.11, CUDA ready environtment

**Pytorch installation**

please refer to [Pytorch official installation](https://pytorch.org/)
```bash
pip install torch torchvision torchaudio
```

**install from requirements.txt**
```bash
cd Spotify_Chatbot
pip install -r requirements.txt
```

**Llama Cpp installation**
We use offline llama2 model to compare with OpenAI API. To use llama2 model, we need to install llama-cpp library.
make sure you have CUDA environment ready
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
pip install pypdf sentence-transformers chromadb langchain
```

### Data preparation
```bash
cd ../data
python data_processing.py
```
The script will download  `SPOTIFY_REVIEWS.csv`, remove unused columns, and save new csv `SPOTIFY_REVIEWS_CLEANED.csv`. This data then loaded into 100 parts, corverted into embeddings, and then saved to local database (Chrome). This process can take hours depends on your hardware.


### Llama2 model download
To prevent long waiting when running the app, you may want to download the llama2 model before running the app
```bash
cd ../llms
python get_llama.py
```

### Run streamlit app
```bash
cd ../app
streamlit run app.py
```



