# Semantic Search, Data: esci-data

## Approach
The semantic search problem is formulated as a Retrieval & Re-Ranking problem, the diagram below shows the building blocks of the system. 
The prototype system is based on models as the result of three steps:
- First step: building the baseline models from the [esci-data dataset repo](https://github.com/amazon-science/esci-data/blob/main/ranking/train.py).
- Second step: using **pre-trained** models from [SentenceTransformers](https://sbert.net/) and [Huggingface](https://huggingface.co/).
  - For Retrieval: bi-encoder models:
    - specific for semantic search: '[sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)'
    - general purpose model: '[sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)'
  - For Ranking: cross-encoder models
    - '[cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)'
    - '[cross-encoder/stsb-roberta-large](https://huggingface.co/cross-encoder/stsb-roberta-large)' (this model didn't work in pre-training and was excluded from the experiment)
- Third step: **fine-tuned** the pre-trained models from step two on the esci-data dataset.

The vector index databases are built using [FAISS](https://github.com/facebookresearch/faiss). 

![Retrieval & Re-Ranking Diagram](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)
Photo credit: [sbert](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)

## Results

### Indivisual Model Performance

| Step        | Model Name                 | Task          | Type          | MRR      | Hits@1 | Hits@5 | Hits@10 | 
| --------    | ----------                 | ----------    | ----------    | ------   | ------ | ------ | ------- | 
| Baseline    | multi-qa-mpnet-base-dot-v1 | Ranking       | Cross-Encoder | 0.8204   | 0.7211 | 0.9447 | 0.9834  | 
| Pre-trained | multi-qa-mpnet-base-dot-v1 | Retrieval     | Bi-Encoder    | 0.7767   | 0.6556 | 0.9320 | 0.9792  |
| Pre-trained | all-mpnet-base-v2          | Retrieval     | Bi-Encoder    | 0.7648   | 0.6426 | 0.9252 | 0.9790  |
| Pre-trained | ms-marco-MiniLM-L-12-v2    | Ranking       | Cross-Encoder | 0.7898   | 0.6738 | 0.9358 | 0.9774  |
| fine-tuned  | multi-qa-mpnet-base-dot-v1 | Retrieval     | Bi-Encoder    | 0.8030   | 0.6970 | 0.9395 | 0.9797  |
| fine-tuned  | all-mpnet-base-v2          | Retrieval     | Bi-Encoder    | 0.7943   | 0.6812 | 0.9378 | 0.9817  |
| fine-tuned  | ms-marco-MiniLM-L-12-v2    | Ranking       | Cross-Encoder | 0.8109   | 0.7067 | 0.9442 | 0.9825  |

### Retrieval & Re-Ranking System Performance
| System based on    | MRR      | Hits@1 | Hits@5 | Hits@10 | 
| --------           | ------   | ------ | ------ | ------- | 
| Pre-trained Models | 0.8969   | 0.8216 | 0.9883 | 0.9883  | 
| Fine-tuned Models  | 0.8349   | 0.7421 | 0.9816 | 1.0000  | 


## Notebooks Descriptions and Considerations:
- EDA and Data Processing, [01_eda_dataprocessing.ipynb]([https://github.com/AliMiraftab/semantic_search_esci/blob/main/Notebooks/01_eda_dataprocessing.ipynb](https://github.com/AliMiraftab/semantic_search_esci/blob/main/notebooks/01_eda_dataprocessing.ipynb)): This notebook provides an overview of the data through general statistics and initial ideas for data processing. It sets the stage for further analysis and feature engineering.
- Feature Engineering, [02_feature_engineering.ipynb](https://github.com/AliMiraftab/semantic_search_esci/blob/main/notebooks/02_feature_engineering.ipynb): In this notebook, I focused on processing text and categorical features. To prepare the data for the Large Language Models (LLMs) used in this prototype, I apply summarization techniques to shorten the features.
  - I employed cleaning practices such as denoising, normalization, and lemmatization on each feature, tailored to the nature of the unstructured data.
  - I explored the LLM model "[Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization)" for text summarization but opted for simpler methods due to time constraints.
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
  - Ultimately, the processed data is ready for the next steps: scoring with pre-trained models and fine-tuning. However, to save time, I only use the 'Product Title' feature as input for the subsequent steps.
- Baseline Models, [03_baseline.ipynb](): proposed by [esci-data train baseline](https://github.com/AliMiraftab/semantic_search_esci/blob/main/notebooks/03_baseline.ipynb).
- Pretrained Modeds, [04_retrieval_ranking_pretrained_models.ipynb](https://github.com/AliMiraftab/semantic_search_esci/blob/main/notebooks/04_retrieval_ranking_pretrained_models.ipynb): using pretrained models for scoring, building the end-to-end Retrieval & Re-Ranking system accordingly, testing, and analysisng the result.
- Fine-Tuned Models [05_retrieval_ranking_pretrained_models.ipynb](https://github.com/AliMiraftab/semantic_search_esci/blob/main/notebooks/05_retrieval_ranking_fine_tuning_models.ipynb): fine-tune the pre-trained models, scoring, building the end-to-end Retrieval & Re-Ranking system accordingly, testing, and analyzing the result.

Note: the prototype is for 'us' language but can be extended to 'es' and 'jp', too.
