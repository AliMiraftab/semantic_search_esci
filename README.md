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



![Retrieval & Re-Ranking Diagram](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)
