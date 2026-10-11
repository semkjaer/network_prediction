# install

python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt

# pipelines 
## for replicability, outputs are precalcuulated and stored

python pipeline.py
python finetuning.py