# install

python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt

# pipelines 
## for replicability, outputs are precalcuulated and stored

python pipeline.py
python finetuning.py
python make_submission.py # creates outputfile at: data/output/submission.csv

# analysis

see prediction.py