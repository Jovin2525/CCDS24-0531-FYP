import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
common_path = os.path.abspath("./")
import argparse

import warnings
warnings.simplefilter('ignore')

from training_w2v_model import training_w2v
from training_model import training_main
from test_model import pred_main
from training_fasttext_model import training_fasttext

def training_deep_model(args):
    """Train a deep learning model using word2vec or FastText embeddings."""
    train_data_path = args.training_file
    val_data_path = args.validation_file
    embedding_model_path = args.embedding_model_file
    out_path = args.out_dir
    model_type = args.model_type
    t_batch = args.training_batch_size
    v_batch = args.validation_batch_size
    lr = args.learning_rate
    max_epoch = args.max_epoch_num
    stop_epoch = args.early_stopping_epoch_num
    threshold = args.threshold
    k_mer = args.k_mer
    seq_max = args.max_len
    
    training_main(
        train_data_path, 
        val_data_path, 
        embedding_model_path, 
        out_path, 
        model_type=model_type,
        t_batch=t_batch, 
        v_batch=v_batch, 
        lr=lr, 
        max_epoch=max_epoch, 
        stop_epoch=stop_epoch, 
        thr=threshold, 
        k_mer=k_mer, 
        seq_max=seq_max
    )
    
def prediction(args):
    """Make predictions using a trained model and word2vec or FastText embeddings."""
    in_path = args.test_file
    out_path = args.out_dir
    embedding_model_path = args.embedding_model_file
    deep_model_path = args.deep_model_file
    model_type = args.model_type
    thresh = args.threshold
    batch_size = args.batch_size
    k_mer = args.k_mer
    seq_max = args.max_len
    vec_ind = args.vec_index
    
    pred_main(
        in_path, 
        out_path, 
        embedding_model_path, 
        deep_model_path, 
        vec_ind,
        thresh, 
        batch_size, 
        k_mer, 
        seq_max,
        model_type=model_type
    )

def training_embedding_model(args):
    """Train a word2vec or FastText embedding model."""
    data_path = args.import_file
    out_path = args.out_dir
    k_mer = args.k_mer
    vector_size = args.vector_size
    window_size = args.window_size
    iteration = args.iteration
    model_type = args.model_type
    
    if model_type == "word2vec":
        training_w2v(data_path, out_path, k_mer, vector_size, window_size, iteration)
    elif model_type == "fasttext":
        training_fasttext(data_path, out_path, k_mer, vector_size, window_size, iteration)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'word2vec' or 'fasttext'")
    
def main():
    """Main CLI entry point."""
    print("Running main..")
    parser = argparse.ArgumentParser(description="Cross-Attention PHV CLI")
    subparsers = parser.add_subparsers(dest="sub_command", help="aphv: this is a CLI to use Cross-Attention PHV")
    subparsers.required = True
    
    # Deep learning model training parser
    deep_train_parser = subparsers.add_parser("train_deep", help="sub-command <train_deep> is used for training a deep learning model for PPI prediction")
    deep_train_parser.add_argument('-t', '--training_file', help='Path of training data file (.csv)', required=True)
    deep_train_parser.add_argument('-v', '--validation_file', help='Path of validation data file (.csv)', required=True)
    deep_train_parser.add_argument('-w', '--embedding_model_file', help='Path of a trained embedding model (word2vec or fasttext)', required=True)
    deep_train_parser.add_argument('-o', '--out_dir', help='Directory to output results', required=True)
    deep_train_parser.add_argument('-m', '--model_type', help='Embedding model type (word2vec or fasttext)', 
                                  choices=['word2vec', 'fasttext'], default='word2vec', type=str)
    deep_train_parser.add_argument('-t_batch', '--training_batch_size', help='Training batch size', default=64, type=int)
    deep_train_parser.add_argument('-v_batch', '--validation_batch_size', help='Validation batch size', default=64, type=int)
    deep_train_parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.0001, type=float)
    deep_train_parser.add_argument('-max_epoch', '--max_epoch_num', help='Maximum epoch number', default=10000, type=int)
    deep_train_parser.add_argument('-stop_epoch', '--early_stopping_epoch_num', help='Epoch number for early stopping', default=20, type=int)
    deep_train_parser.add_argument('-thr', '--threshold', help='Threshold to determined whether interact or not', default=0.5, type=float)
    deep_train_parser.add_argument('-k_mer', '--k_mer', help='Size of k in k_mer', default=2, type=int)
    deep_train_parser.add_argument('-max_len', '--max_len', help='Maximum sequence length', default=9000, type=int)
    deep_train_parser.set_defaults(handler=training_deep_model)
    # Training deep models (Examples - change paths if necessary)
    # With word2vec embeddings:
    # python main.py train_deep -t ./data/train.csv -v ./data/val.csv -w ./embedding_models/word2vec_model.pt -o ./deep_model -m word2vec
    # With FastText embeddings:
    # python main.py train_deep -t ./data/train.csv -v ./data/val.csv -w ./embedding_models/fasttext_model.pt -o ./deep_model -m fasttext
    
    # Embedding model training parser
    embedding_train_parser = subparsers.add_parser("train_embedding", help="sub-command <train_embedding> is used for training an embedding model (word2vec or fasttext) to encode amino acid sequences")
    embedding_train_parser.add_argument("-i", "--import_file", help="Path of training data (.csv)", required=True)
    embedding_train_parser.add_argument("-o", "--out_dir", help="Directory to save embedding model", required=True)
    embedding_train_parser.add_argument("-m", "--model_type", help="Embedding model type to train", 
                                     choices=['word2vec', 'fasttext'], default='word2vec', type=str)
    embedding_train_parser.add_argument("-k_mer", "--k_mer", help='Size of k in k_mer', default=2, type=int)
    embedding_train_parser.add_argument("-v_s", "--vector_size", help="Vector size", default=128, type=int)
    embedding_train_parser.add_argument("-w_s", "--window_size", help="Window size", default=5, type=int)
    embedding_train_parser.add_argument("-iter", "--iteration", type=int, default=50, help="Iteration of training")
    embedding_train_parser.set_defaults(handler=training_embedding_model)
    # Training embedding models (Examples - change paths if necessary)
    # Training word2vec (default k-mer=2):
    # python main.py train_embedding -i ./data/w2v_data.csv -o ./embedding_models -m word2vec
    # Training FastText:
    # python main.py train_embedding -i ./data/w2v_data.csv -o ./embedding_models -m fasttext -k_mer 4
    
    # Prediction parser
    pred_parser = subparsers.add_parser("predict", help="sub-command <predict> is used for prediction of PPI")
    pred_parser.add_argument('-t', '--test_file', help='Path of data file (.csv)', required=True)
    pred_parser.add_argument('-o', '--out_dir', help='Directory to output results', required=True)
    pred_parser.add_argument('-w', '--embedding_model_file', help='Path of a trained embedding model (word2vec or fasttext)', required=True)
    pred_parser.add_argument('-d', '--deep_model_file', help='Path of a trained aphv model', required=True)
    pred_parser.add_argument('-m', '--model_type', help='Embedding model type (word2vec or fasttext)', 
                           choices=['word2vec', 'fasttext'], default='word2vec', type=str)
    pred_parser.add_argument('-vec', '--vec_index', help='Flag whether features output', action='store_true', default=True)
    pred_parser.add_argument('-thr', '--threshold', help='Threshold to determined whether interact or not', default=0.5, type=float)
    pred_parser.add_argument('-batch', '--batch_size', help='Batch size', default=64, type=int)
    pred_parser.add_argument('-k_mer', '--k_mer', help='Size of k in k_mer', default=2, type=int)
    pred_parser.add_argument('-max_len', '--max_len', help='Maximum sequence length', default=9000, type=int)
    pred_parser.set_defaults(handler=prediction)
    # Making predictions (Examples - change paths if necessary)
    # With word2vec embeddings:
    # python main.py predict -t ./data/test.csv -w ./embedding_models/word2vec_model.pt -d ./deep_model/data_model/deep_model.pth -o ./results -m word2vec
    # With FastText embeddings:
    # python main.py predict -t ./data/test.csv -w ./embedding_models/fasttext_model.pt -d ./deep_model/data_model/deep_model.pth -o ./results -m fasttext

    # Parse arguments and call the appropriate handler
    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)

if __name__ == "__main__":
    main()