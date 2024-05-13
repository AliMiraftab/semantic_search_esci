# Semantic Search, Data: esci-data

## Approach
The semantic search problem is formulated as a Retrieval & Re-Ranking problem, the diagram below shows the building blocks of the system. 
The prototype system is based on models as the result of three steps:
- First step: building the baseline models from the esci-data dataset repo.
- Second step: using **pre-trained** models from [SentenceTransformers](https://sbert.net/) and [Huggingface](https://huggingface.co/).
  - For Retrieval: bi-encoder models:
    - specific for semantic search: 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    - general purpose model: 'sentence-transformers/all-mpnet-base-v2'
  - For Ranking: cross-encoder models
    - 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    - 'cross-encoder/stsb-roberta-large' (this model didn't work in pre-training and was excluded from the experiment)
- Third step: **fine-tuned** the pre-trained models from step two on the esci-data dataset.

The vector index databases are built using [FAISS](https://github.com/facebookresearch/faiss). 

Notebooks descriptions, and considerations:
- 01_eda_dataprocessing (EDA and Data Processing): includes general stats of the data and primary ideas for processing the data.
- 02_feature_engineering (Feature Engineering): 



![Retrieval & Re-Ranking Diagram](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)
