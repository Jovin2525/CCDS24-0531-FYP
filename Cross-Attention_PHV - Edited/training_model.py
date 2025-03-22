import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from torch import optim
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn
import numpy as np
from Bio import SeqIO
from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from deep_network import Transformer_PHV
import warnings
warnings.simplefilter('ignore')
import joblib
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
import pickle
import h5py
import sys


metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

amino_asids_vector = np.eye(20)
normal_amino_asids = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    return data


def load_joblib(filename):
    with open(filename, "rb") as f:
        data = joblib.load(f)
    return data

def save_joblib(filename, data):
    with open(filename, "wb") as f:
        joblib.dump(data, f, compress = 3)

def w2v_encoding(seq, model, window, step, k_mer):
    vecs = []
    for i in range(0, len(seq), step):
        pep = seq[i: i+ window]
        if(len(pep) >= k_mer):
            vec = np.mean([model.wv[pep[j: j + k_mer]] for j in range(len(pep) - k_mer + 1)], axis = 0)
        vecs.append(vec)
        
    return vecs

def d2v_encoding(seq, model, window, step, k_mer):
    vecs = []
    for i in range(0, len(seq), step):
        pep = seq[i: i+ window]
        vec = model.infer_vector([pep[j: j + k_mer] for j in range(len(pep) - k_mer + 1)])
        vecs.append(vec)
        
    return vecs

def transform_seq_scale(seq, stride):
    if(seq%stride == 0):
        return int(seq/stride)
    else:
        return int(seq/stride) + 1

class antibody_data_sets(data.Dataset):
    def __init__(self, data_sets, enc_model, enc_seq_max = 9000, window = 20, stride = 10, k_mer = 1):
        super().__init__()
        self.heavy_chain = data_sets["heavy_chain"].values.tolist()
        self.light_chain = data_sets["light_chain"].values.tolist()
        self.y = np.array(data_sets["paired"].values.tolist()).reshape([len(data_sets["paired"]),1])
        self.enc_model = enc_model
        self.enc_seq_max = enc_seq_max
        self.window, self.stride = window, stride
        self.k_mer = k_mer

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light = encoding_antibody_chains(self.heavy_chain[idx], self.light_chain[idx], self.enc_model, enc_seq_max = self.enc_seq_max, window = self.window, stride = self.stride, k_mer = self.k_mer)
        return heavy_chain_mat.to(device), light_chain_mat.to(device), attn_mask_heavy.to(device), attn_mask_light.to(device), torch.tensor(self.y[idx], device=device, dtype=torch.float)
    
def create_mat_dict_w2v(seqs, enc_model, k_mer, filepath="encoded_sequences.pkl"):
    if os.path.exists(filepath):
        print(f"Loading encoded sequences from {filepath}")
        with open(filepath, "rb") as f:
            seq2mat_dict = pickle.load(f)
    else:
        print("Encoding sequences...")
        seqs = list(set(seqs))  # Remove duplicates
        seq2mat_dict = {}
        
        for i in range(len(seqs)):
            seq2mat_dict[seqs[i]] = torch.tensor([enc_model.wv[seqs[i][j: j + k_mer]] for j in range(len(seqs[i]) - k_mer + 1)])

        # Save the dictionary to a file using pickle - uncomment to save embedding -> 80+gb
        with open(filepath, "wb") as f:
            print(f"Dumping to {filepath}")
            pickle.dump(seq2mat_dict, f)

    return seq2mat_dict

class DeepNet():
    def __init__(self, out_path, enc_dict, model_params, training_params, encoding_params):
        self.out_path = out_path
        self.enc_model = enc_dict
        self.model_params = model_params
        self.tra_batch_size = training_params["training_batch_size"]
        self.val_batch_size = training_params["validation_batch_size"]
        self.lr = training_params["lr"]
        self.enc_seq_max_train = encoding_params["enc_seq_max_train"]
        self.enc_seq_max_val = encoding_params["enc_seq_max_val"]
        self.max_epoch = training_params["max_epoch"]
        self.early_stop = training_params["early_stopping"]
        self.thresh = training_params["thresh"]
        self.k_mer = encoding_params["k_mer"]
        self.stopping_met = training_params["stopping_met"]

    def compute_pos_weight(self, train_loader):
        n_positive = 0
        n_total = 0
        for _, _, _, _, labels in train_loader:
            n_positive += labels.sum().item()
            n_total += len(labels)
        # Keep pos_weight on CPU initially
        return torch.tensor([n_total/n_positive - 1])

    def model_training(self, train_data_sets, val_data_sets, load_model_path=None):
        os.makedirs(self.out_path + "/data_model", exist_ok=True)
        os.makedirs(self.out_path + "/logs", exist_ok=True)
        log_file = open(self.out_path + "/logs/training.log", "w")
        sys.stdout = log_file

        # Create data loaders
        tra_data_all = antibody_data_sets(train_data_sets, enc_model=self.enc_model, 
                                        enc_seq_max=self.enc_seq_max_train, 
                                        window=self.model_params["kernel_size"], 
                                        stride=self.model_params["stride"], 
                                        k_mer=self.k_mer)
        train_loader = DataLoader(dataset=tra_data_all, batch_size=self.tra_batch_size, shuffle=True)

        val_data_all = antibody_data_sets(val_data_sets, enc_model=self.enc_model, 
                                        enc_seq_max=self.enc_seq_max_val, 
                                        window=self.model_params["kernel_size"], 
                                        stride=self.model_params["stride"], 
                                        k_mer=self.k_mer)
        val_loader = DataLoader(dataset=val_data_all, batch_size=self.val_batch_size, shuffle=True)
        
        # Initialize model on GPU
        self.model = Transformer_PHV(
            filter_num=self.model_params["filter_num"],
            kernel_size_w2v=self.model_params["kernel_size"],
            stride_w2v=self.model_params["stride"],
            n_heads=self.model_params["n_heads"],
            d_dim=self.model_params["d_dim"],
            feature=self.model_params["feature"],
            pooling_dropout=self.model_params["pooling_dropout"],
            linear_dropout=self.model_params["linear_dropout"]
        ).to(device)
        
        # Load model state if provided
        if load_model_path and os.path.exists(load_model_path):
            checkpoint = torch.load(load_model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            max_met = checkpoint['max_met']
            early_stop_count = checkpoint['early_stop_count']
            print(f"Loaded model state from {load_model_path}, resuming from epoch {start_epoch}", flush=True)
        else:
            start_epoch = 0
            max_met = 100
            early_stop_count = 0
            if load_model_path:
                print(f"Model path {load_model_path} does not exist, starting from epoch {start_epoch}", flush=True)

        # Initialize optimizer and loss function
        self.opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
        pos_weight = self.compute_pos_weight(train_loader).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in range(start_epoch, self.max_epoch):
            training_losses, validation_losses = [], []
            train_probs, val_probs = [], []
            train_labels, val_labels = [], []
            self.model.train()

            # Training loop
            for i, (heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light, labels) in enumerate(train_loader):
                # Move batch to GPU
                heavy_chain_mat = heavy_chain_mat.to(device)
                light_chain_mat = light_chain_mat.to(device)
                attn_mask_heavy = attn_mask_heavy.to(device)
                attn_mask_light = attn_mask_light.to(device)
                labels = labels.to(device)

                self.opt.zero_grad()
                probs = self.model(heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light)
                
                loss = self.criterion(probs, labels)
                loss.backward()
                self.opt.step()

                # Store results on CPU
                training_losses.append(loss.item())
                train_probs.extend(probs.cpu().detach().numpy().flatten().tolist())
                train_labels.extend(labels.cpu().detach().numpy().astype('int32').flatten().tolist())

                # Clear GPU memory
                del heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light, labels, probs, loss
                torch.cuda.empty_cache()

            # Process training metrics on CPU
            train_probs_tensor = torch.tensor(train_probs).float()
            train_labels_tensor = torch.tensor(train_labels).float()
            loss_epoch = self.criterion(train_probs_tensor.to(device), train_labels_tensor.to(device))
            print("=============================", flush=True)
            print("Training loss:: " + str(loss_epoch.item()), flush=True)

            # Calculate metrics on CPU
            for key in metrics_dict.keys():
                if key not in ["auc", "AUPRC"]:
                    metrics = metrics_dict[key](train_labels, train_probs, thresh=self.thresh)
                else:
                    metrics = metrics_dict[key](train_labels, train_probs)
                print("train_" + key + ": " + str(metrics), flush=True)

            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_probs, thresh=self.thresh)
            print(f"train_true_negative:: value: {tn_t}, epoch: {epoch + 1}", flush=True)
            print(f"train_false_positive:: value: {fp_t}, epoch: {epoch + 1}", flush=True)
            print(f"train_false_negative:: value: {fn_t}, epoch: {epoch + 1}", flush=True)
            print(f"train_true_positive:: value: {tp_t}, epoch: {epoch + 1}", flush=True)
            print("-----------------------------", flush=True)

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                for i, (heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light, labels) in enumerate(val_loader):
                    # Move batch to GPU
                    heavy_chain_mat = heavy_chain_mat.to(device)
                    light_chain_mat = light_chain_mat.to(device)
                    attn_mask_heavy = attn_mask_heavy.to(device)
                    attn_mask_light = attn_mask_light.to(device)
                    labels = labels.to(device)

                    probs = self.model(heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light)
                    loss = self.criterion(probs, labels)

                    # Store results on CPU
                    validation_losses.append(loss.item())
                    val_probs.extend(probs.cpu().numpy().flatten().tolist())
                    val_labels.extend(labels.cpu().numpy().astype('int32').flatten().tolist())

                    # Clear GPU memory
                    del heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light, labels, probs, loss
                    torch.cuda.empty_cache()

            # Process validation metrics on CPU
            val_probs_tensor = torch.tensor(val_probs).float()
            val_labels_tensor = torch.tensor(val_labels).float()
            loss_epoch = self.criterion(val_probs_tensor.to(device), val_labels_tensor.to(device))
            print("Validation loss:: " + str(loss_epoch.item()), flush=True)

            # Calculate validation metrics on CPU
            for key in metrics_dict.keys():
                if key not in ["auc", "AUPRC"]:
                    metrics = metrics_dict[key](val_labels, val_probs, thresh=self.thresh)
                else:
                    metrics = metrics_dict[key](val_labels, val_probs)
                print("validation_" + key + ": " + str(metrics), flush=True)

            # Early stopping logic
            if self.stopping_met == "loss":
                epoch_met = loss_epoch.item()
            else:
                epoch_met = 1 - metrics_dict[self.stopping_met](val_labels, val_probs)

            tn_v, fp_v, fn_v, tp_v = cofusion_matrix(val_labels, val_probs, thresh=self.thresh)
            print(f"validation_true_negative:: value: {tn_v}, epoch: {epoch + 1}", flush=True)
            print(f"validation_false_positive:: value: {fp_v}, epoch: {epoch + 1}", flush=True)
            print(f"validation_false_negative:: value: {fn_v}, epoch: {epoch + 1}", flush=True)
            print(f"validation_true_positive:: value: {tp_v}, epoch: {epoch + 1}", flush=True)

            # Save best model
            if epoch_met < max_met:
                early_stop_count = 0
                max_met = epoch_met
                current_dir = os.getcwd()
                os.makedirs(self.out_path + "/data_model", exist_ok=True)
                os.chdir(self.out_path + "/data_model")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'max_met': max_met,
                    'early_stop_count': early_stop_count
                }, "deep_model.pth")
                os.chdir(current_dir)
                final_val_probs = val_probs
                final_val_labels = val_labels
                final_train_probs = train_probs
                final_train_labels = train_labels
            else:
                early_stop_count += 1
                if early_stop_count >= self.early_stop:
                    print(f'Training cannot improve from epoch {epoch + 1 - self.early_stop}\tBest {self.stopping_met}: {max_met}', flush=True)
                    break

        # Final metrics
        print(self.thresh, flush=True)
        for key in metrics_dict.keys():
            if key not in ["auc", "AUPRC"]:
                train_metrics = metrics_dict[key](final_train_labels, final_train_probs, thresh=self.thresh)
                val_metrics = metrics_dict[key](final_val_labels, final_val_probs, thresh=self.thresh)
            else:
                train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
            print(f"train_{key}: {train_metrics}", flush=True)
            print(f"test_{key}: {val_metrics}", flush=True)

        threshold_1, threshold_2 = cutoff(final_val_labels, final_val_probs)
        print(f"Best threshold (AUC) is {threshold_1}")
        print(f"Best threshold (PRC) is {threshold_2}")
        log_file.close()

        return ""
    
def encoding_antibody_chains(heavy_chain_seq, light_chain_seq, enc_model, enc_seq_max, window, stride, k_mer):
    # Move the encoded sequences to the GPU
    heavy_chain_mat = enc_model[heavy_chain_seq].to(device)
    light_chain_mat = enc_model[light_chain_seq].to(device)
    
    # Sequence lengths
    mat_len_heavy, mat_len_light = len(heavy_chain_mat), len(light_chain_mat)
    w2v_seq_max = enc_seq_max - k_mer + 1
    
    # Pad heavy and light chain matrices to ensure they fit the maximum sequence length
    heavy_chain_mat = torch.nn.functional.pad(heavy_chain_mat, (0, 0, 0, w2v_seq_max - mat_len_heavy)).float().to(device)
    light_chain_mat = torch.nn.functional.pad(light_chain_mat, (0, 0, 0, w2v_seq_max - mat_len_light)).float().to(device)
    
    # Calculate the convolutional output lengths
    mat_conv_len_heavy = max(int((mat_len_heavy - window) / stride) + 1, 1)
    mat_conv_len_light = max(int((mat_len_light - window) / stride) + 1, 1)
    
    # Maximum convolution length based on sequence size and convolution parameters
    max_conv_len = int((w2v_seq_max - window) / stride) + 1

    # Attention masks for heavy and light chains, padded and moved to GPU
    w2v_attn_mask_heavy = torch.cat((
        torch.zeros((mat_conv_len_heavy, max_conv_len), device=device).long(),
        torch.ones((max_conv_len - mat_conv_len_heavy, max_conv_len), device=device).long()
    )).transpose(-1, -2).bool()
    
    w2v_attn_mask_light = torch.cat((
        torch.zeros((mat_conv_len_light, max_conv_len), device=device).long(),
        torch.ones((max_conv_len - mat_conv_len_light, max_conv_len), device=device).long()
    )).transpose(-1, -2).bool()
    
    # Move tensors back to the CPU before returning
    heavy_chain_mat = heavy_chain_mat.cpu()
    light_chain_mat = light_chain_mat.cpu()
    w2v_attn_mask_heavy = w2v_attn_mask_heavy.cpu()
    w2v_attn_mask_light = w2v_attn_mask_light.cpu()
    
    return heavy_chain_mat, light_chain_mat, w2v_attn_mask_heavy, w2v_attn_mask_light


def training_main(train_path, val_path, w2v_path, out_path, t_batch = 32, v_batch = 32, lr = 0.0001, max_epoch = 10000, stop_epoch = 20, thr = 0.5, k_mer = 4, seq_max = 9000):
    print("Setting parameters", flush = True)
    model_params = {"filter_num": 128, "kernel_size": 20, "stride": 10, "n_heads": 4, "d_dim": 32, "feature": 128, "pooling_dropout": 0.5, "linear_dropout": 0.3}
    training_params = {"training_batch_size": t_batch, "validation_batch_size": v_batch, "lr": lr, "early_stopping": stop_epoch, "max_epoch": max_epoch, "thresh": thr, "stopping_met": "auc"}
    encoding_params = {"enc_seq_max_train": seq_max, "enc_seq_max_val": seq_max, "k_mer": k_mer}

    print("Loading datasets", flush = True)
    training_data = file_input_csv(train_path)
    validation_data = file_input_csv(val_path)

    print("Loading word2vec model", flush = True)
    w2v_model = word2vec.Word2Vec.load(w2v_path)

    print("Encoding amino acid sequences", flush = True)
    embedding_path = "./encoded_sequences.pkl"
    mat_dict = create_mat_dict_w2v(training_data["heavy_chain"].values.tolist() + validation_data["heavy_chain"].values.tolist() + training_data["light_chain"].values.tolist() + validation_data["light_chain"].values.tolist(), w2v_model, encoding_params["k_mer"],embedding_path)

    print("Start training a deep neural network model", flush = True)
    net = DeepNet(out_path, mat_dict, model_params, training_params, encoding_params)
    load_model_path = out_path + "/data_model/deep_model.pth"
    out = net.model_training(training_data, validation_data, load_model_path)

    print("Finish processing", flush = True)





































