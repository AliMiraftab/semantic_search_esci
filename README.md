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

![Retrieval & Re-Ranking Diagram](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)

## Results

| Step        | Model Name | Task       | Type          | MRR | Hits@1 | Hits@5 | Hits@10 | 
| --------    | ---------- | ---------- | ----------    |     |        |        |         | 
| Baseline    |            | Ranking    | Cross-Encoder |     |        |        |         |  
| Pre-trained |            | Retrieval  | Bi-Encoder    |     |        |        |         |  
| Pre-trained |            | Retrieval  | Bi-Encoder    |     |        |        |         |  

## Notebooks Descriptions and Considerations:
- 01_eda_dataprocessing.ipynb (EDA and Data Processing): This notebook provides an overview of the data through general statistics and initial ideas for data processing. It sets the stage for further analysis and feature engineering.
- 02_feature_engineering.ipynb (Feature Engineering): In this notebook, we focus on processing text and categorical features. To prepare the data for the Large Language Models (LLMs) used in this prototype, we apply summarization techniques to shorten the features.
  - We employ cleaning practices such as denoising, normalization, and lemmatization on each feature, tailored to the nature of the unstructured data.
  - We explore the LLM model "Falconsai/text_summarization" for text summarization but opt for simpler methods due to time constraints.
  - The final feature is a combination of all feature columns:
    ```python
    def combine_features(row):
      combined = ""
      combined += f"Brand: {row[col_product_brand]}, " \
                  f"Color: {row[col_product_color]}, " \
                  f"Product Title: {row[col_product_title]}, " \
                  f"Product Description: {row[col_product_description]}, and " \
                  f"Product Features: {row[col_product_bullet_point]}."
      return combined
    ```
  - Ultimately, the processed data is ready for the next steps: scoring with pre-trained models and fine-tuning. However, to save time, we only use the 'Product Title' feature as input for the subsequent steps.
- 03_baseline.ipynb (Baseline Models): proposed by [esci-data train baseline](https://github.com/amazon-science/esci-data/blob/main/ranking/train.py).
- 04_retrieval_ranking_pretrained_models.ipynb: using pretrained models for scoring, building the end-to-end Retrieval & Re-Ranking system accordingly, testing, and analysisng the result.
- 05_retrieval_ranking_pretrained_models.ipynb: fine-tune the pretrained models, scoring, building the end-to-end Retrieval & Re-Ranking system accordingly, testing, and analysisng the result.
