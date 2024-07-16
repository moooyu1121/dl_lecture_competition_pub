python -m venv .venv
call .venv\Scripts\activate
.venv\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pause