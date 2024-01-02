import os
from urllib.request import urlretrieve

model_path = "llama-2-7b-chat.Q4_0.gguf"
if not os.path.isfile(model_path):
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf?download=true"
    filename = "llama-2-7b-chat.Q4_0.gguf"
    urlretrieve(url, filename)
