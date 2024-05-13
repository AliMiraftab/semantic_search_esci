# Semantic Search, Data: esci-data

## Approach
The semantic search problem is formulated as a Retrieval & Re-Ranking problem, the diagram below shows the building blocks of the system. 
The prototype system is based on models as the result of three steps:
- First step: building the baseline models from the esci-data dataset repo.
- Second step: using pre-trained models from [SentenceTransformers](https://sbert.net/) and [Huggingface](https://huggingface.co/).
  - For Retrieval: bi-encoder models
  - For Ranking: cross-encoder models
- Third step: fine-tuned the pre-trained model
  - For Retrieval: bi-encoder models
  - For Ranking: cross-encoder models
 



![Retrieval & Re-Ranking Diagram](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)
