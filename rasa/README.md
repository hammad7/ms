## List env
conda info --envs

## Delete existing env
conda remove -n rasa --all

conda create -n rasa python==3.7.9
conda activate rasa
pip install rasa

## connect venv to Jupyter notebook (Optional)
pip install ipykernel
python -m ipykernel install --user --name rasa --display-name rasa


## Install compatible spacy 
pip install rasa[spacy]==2.8.2
## Download spacy model
python -m spacy download en_core_web_md
