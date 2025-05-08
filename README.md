# CCDS24-0531-FYP

1. Data preprocessing is handled in the file "data_processing.ipynb", in which random shuffling, dissimilarity-based negative sampling, and motif-based approach is used to generate the training data.
2. VAE generation is handled in the folder "vae", wrapper.py is the main file for VAE generation.
3. plot_dist.ipynb is used to plot the distribution of the final antibody datasets
4. umap.ipynb is used to plot the UMAP of the embedding space encoded by Word2Vec/FastText - this should be done after Word2Vec/FastText has encoded the relevant sequences during model training
5. training_w2v_model.py is used to train a Word2Vec model eg. `python main.py train_embedding -i ./data/w2v_data.csv -o ./embedding_models -m word2vec`
6. training_fasttext_model.py is used to train a FastText model eg. `python main.py train_embedding -i ./data/w2v_data.csv -o ./embedding_models -m fasttext`
7. training_model.py is used to train a deep learning model for antibody prediction eg. `python main.py train_deep -t ./data/train_agg.csv -v ./data/val_agg.csv -w ./embedding_models/[embedding_model] -o ./deep_model -m [model_type]`
8. test_model.py is used to test the trained model eg. `python main.py predict -t ./data/test_agg.csv -w ./embedding_models/[embedding_model].pt -d ./deep_model/[deep_model].pth -o ./results -m [model_type]`
9. To run the web application, simply run `python app.py` after ensuring you are in the 'Cross-Attention_PHV - Edited' folder

- w2v_data.csv (used to train both embedding models), train_agg.csv, val_agg.csv and test_agg.csv are found in 'Cross-Attention_PHV - Edited/data'. train_agg.csv, val_agg.csv and test_agg.csv represent the final dataset with negative samples generated via dissimilarity-based negative sampling, motif-based approach and VAE.

Note: Cross_Attention_PHV is adapted from Sho and Kurata's 2022 paper: 'Cross-attention PHV: Prediction of human and virus protein-protein interactions using cross-attention–based neural networks'. Their github link is https://github.com/kuratahiroyuki/Cross-Attention_PHV
