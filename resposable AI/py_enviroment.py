# 1 install libraries

import os
import sys

print(f"Using Python {sys.version}")
'''
os.system(f"/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pip install --upgrade pip setuptools wheel")
os.system(f"/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m pip install sentencepiece safetensors diffusers accelerate transformers")
os.system(f"/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 PixArt.py")

# Upgrade pip, setuptools, wheel
os.system(f"{sys.executable} -m pip install --upgrade pip setuptools wheel")

# Install PyTorch (CPU/MPS for Mac, CUDA for NVIDIA if needed)
# If on Mac, you can skip the --extra-index-url line for CUDA
os.system(f"{sys.executable} -m pip install torch torchvision torchaudio")

# Install Hugging Face + dependencies
os.system(f"{sys.executable} -m pip install diffusers==0.30.0 transformers accelerate safetensors")

# Verify install
os.system(f"{sys.executable} -m pip show diffusers")


os.system('pip install diffusers transformers accelerate safetensors')
os.system('pip install torch --extra-index-url https://download.pytorch.org/whl/cu118')
os.system('pip install torch torchvision torchaudio')

os.system('pip install torch --extra-index-url https://download.pytorch.org/whl/cu118')
os.system('pip show diffusers')



os.system('pip install "numpy<2.0.0"')
os.system('pip install --upgrade pip')


os.system('pip install --upgrade diffusers transformers accelerate safetensors')
os.system('pip install tiktoken')
os.system('pip install blobfile')
os.system('pip install protobuf')
'''
#os.system('python -m pip install -U sentencepiece transformers diffusers accelerate safetensors')

#os.system('pip install beautifulsoup4')
#os.system('pip install python-dotenv')
#os.system('pip freeze')

#os.system('python3 -m pip install -U diffusers transformers accelerate peft safetensors datasets')
# (Mac MPS tips)
#export PYTORCH_ENABLE_MPS_FALLBACK=1

os.system('pip install accelerate peft datasets')



