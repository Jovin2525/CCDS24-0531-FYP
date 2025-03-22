# CCDS24-0531-FYP

1. Data preprocessing is handled in the file "data_processing.ipynb", in which random shuffling, dissimilarity-based negative sampling, and motif-based approach is used to generate the training data.
2. VAE generation is handled in the folder "vae", wrapper.py is the main file for VAE generation.
3. plot_dist.ipynb is used to plot the distribution of the generated antibody datasets
4. umap.ipynb is used to plot the UMAP of the embedding space encoded by Word2Vec/FastText
5. training_w2v_model.py is used to train a Word2Vec model eg. `python main.py train_w2v -i1 ./data/w2v_data.csv -o ./w2v_model`
6. training_fasttext_model.py is used to train a FastText model eg. `python main.py train_fasttext -i1 ./data/w2v_data.csv -o ./w2v_model`
7. training_model.py is used to train a deep learning model for antibody prediction eg. `python main.py train_deep -t ./data/train_agg.csv -v ./data/val_agg.csv -w ./w2v_model/AA_model_fasttext.pt -o ./deep_model`
8. test_model.py is used to test the trained model eg. `python main.py predict -i ./data/test_agg.csv -o ./deep_model/test -w ./w2v_model/AA_model_fasttext.pt -d ./deep_model/data_model/deep_model.pth`