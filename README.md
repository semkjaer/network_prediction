

# install

python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt

# pipelines 

## for replicability, outputs are precalcuulated and stored so theres no need to run these unless changes are made

python pipeline.py
python finetuning.py
python make_submission.py # creates outputfile at: data/output/submission.csv

# analysis

run all on prediction.ipynb and inspect the output of the last cell to see the analaysis