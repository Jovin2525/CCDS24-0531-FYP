from flask import Flask, render_template, request, jsonify, current_app
import os
import torch
import numpy as np
import sys
import logging
import traceback

try:
    from gensim.models import word2vec, FastText
    import pickle
    from deep_network import Transformer_PHV
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure all required modules are installed and accessible")
    sys.exit(1)

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Global variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device: {device}")

embedding_model = None
deep_model = None
embedding_dict = {}
model_params = {}
encoding_params = {}
model_type = "word2vec"

def load_embedding_model(model_path, model_type="word2vec"):
    """Load either word2vec or FastText model"""
    app.logger.info(f"Loading {model_type} model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    try:
        if model_type.lower() == "word2vec":
            return word2vec.Word2Vec.load(model_path)
        elif model_type.lower() == "fasttext":
            return FastText.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'word2vec' or 'fasttext'")
    except Exception as e:
        app.logger.error(f"Error loading {model_type} model: {str(e)}")
        raise

def encode_sequence(seq, model_type="word2vec"):
    """Encode a sequence using the loaded embedding model"""
    global embedding_model, embedding_dict
    
    # Check if sequence is already in dictionary
    if seq in embedding_dict:
        return embedding_dict[seq]
    
    # Encode new sequence
    k_mer = encoding_params["k_mer"]
    
    try:
        # Collect vectors in a list first
        vectors = []
        for j in range(len(seq) - k_mer + 1):
            k_mer_seq = seq[j: j + k_mer]
            try:
                vectors.append(embedding_model.wv[k_mer_seq])
            except KeyError:
                # Handle OOV for word2vec (FastText should handle automatically)
                if model_type.lower() == "word2vec":
                    app.logger.warning(f"K-mer not found in word2vec model: {k_mer_seq}")
                    vector_size = embedding_model.vector_size
                    vectors.append(np.zeros(vector_size))
                else:
                    raise
        
        # Convert list of arrays to a single numpy array first, then to tensor
        tensor = torch.tensor(np.array(vectors))
        embedding_dict[seq] = tensor
        return tensor
    except Exception as e:
        app.logger.error(f"Error encoding sequence: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

def encoding_antibody_chains(heavy_chain_seq, light_chain_seq):
    """Encode heavy and light chains for model input"""
    # Get encoding parameters
    enc_seq_max = encoding_params["enc_seq_max"]
    window = model_params["kernel_size"]
    stride = model_params["stride"]
    k_mer = encoding_params["k_mer"]
    
    try:
        # Encode sequences
        heavy_chain_mat = encode_sequence(heavy_chain_seq, model_type).to(device)
        light_chain_mat = encode_sequence(light_chain_seq, model_type).to(device)
        
        # Calculate sequence lengths
        mat_len_heavy, mat_len_light = len(heavy_chain_mat), len(light_chain_mat)
        w2v_seq_max = enc_seq_max - k_mer + 1
        
        # Pad sequences
        heavy_chain_mat = torch.nn.functional.pad(
            heavy_chain_mat, 
            (0, 0, 0, w2v_seq_max - mat_len_heavy)
        ).float()
        
        light_chain_mat = torch.nn.functional.pad(
            light_chain_mat, 
            (0, 0, 0, w2v_seq_max - mat_len_light)
        ).float()
        
        # Calculate convolution lengths
        mat_conv_len_heavy = max(int((mat_len_heavy - window) / stride) + 1, 1)
        mat_conv_len_light = max(int((mat_len_light - window) / stride) + 1, 1)
        max_conv_len = int((w2v_seq_max - window) / stride) + 1

        # Create attention masks
        w2v_attn_mask_heavy = torch.cat((
            torch.zeros((mat_conv_len_heavy, max_conv_len), device=device).long(),
            torch.ones((max_conv_len - mat_conv_len_heavy, max_conv_len), device=device).long()
        )).transpose(-1, -2).bool()
        
        w2v_attn_mask_light = torch.cat((
            torch.zeros((mat_conv_len_light, max_conv_len), device=device).long(),
            torch.ones((max_conv_len - mat_conv_len_light, max_conv_len), device=device).long()
        )).transpose(-1, -2).bool()
        
        return heavy_chain_mat, light_chain_mat, w2v_attn_mask_heavy, w2v_attn_mask_light
    except Exception as e:
        app.logger.error(f"Error encoding antibody chains: {str(e)}")
        raise

def initialize_models(embedding_path, model_path, model_type="word2vec", k_mer=2, seq_max=9000):
    """Initialize the embedding and deep models"""
    global embedding_model, deep_model, model_params, encoding_params
    
    app.logger.info(f"Initializing models: embedding={embedding_path}, model={model_path}, type={model_type}")
    
    try:
        # Set parameters
        model_params = {
            "filter_num": 128,
            "kernel_size": 20,
            "stride": 10,
            "n_heads": 4,
            "d_dim": 32,
            "feature": 128,
            "pooling_dropout": 0.5,
            "linear_dropout": 0.3
        }
        
        encoding_params = {
            "enc_seq_max": seq_max,
            "k_mer": k_mer
        }
        
        # Load embedding model
        embedding_model = load_embedding_model(embedding_path, model_type)
        app.logger.info(f"Embedding model loaded successfully: {type(embedding_model)}")
        
        # Initialize deep model
        deep_model = Transformer_PHV(
            filter_num=model_params["filter_num"],
            kernel_size_w2v=model_params["kernel_size"],
            stride_w2v=model_params["stride"],
            n_heads=model_params["n_heads"],
            d_dim=model_params["d_dim"],
            feature=model_params["feature"],
            pooling_dropout=model_params["pooling_dropout"],
            linear_dropout=model_params["linear_dropout"]
        ).to(device)
        
        # Load deep model weights
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            deep_model.load_state_dict(checkpoint['model_state_dict'])
            app.logger.info("Model loaded from checkpoint format")
        else:
            deep_model.load_state_dict(checkpoint)
            app.logger.info("Model loaded from direct state dict format")
            
        deep_model.eval()
        app.logger.info("Deep model loaded and set to eval mode")
        return True
    except Exception as e:
        app.logger.error(f"Error initializing models: {str(e)}")
        app.logger.error(traceback.format_exc())
        return False

def predict_pairing(heavy_chain, light_chain):
    """Make prediction on a pair of sequences"""
    global deep_model
    
    if deep_model is None:
        return {"error": "Model not initialized"}
    
    try:
        # Encode sequences
        heavy_chain_mat, light_chain_mat, attn_mask_heavy, attn_mask_light = encoding_antibody_chains(
            heavy_chain, light_chain
        )
        
        # Make prediction
        with torch.no_grad():
            prob = deep_model(
                heavy_chain_mat.unsqueeze(0), 
                light_chain_mat.unsqueeze(0),
                attn_mask_heavy.unsqueeze(0),
                attn_mask_light.unsqueeze(0)
            )
            score = prob.cpu().detach().item()
            
        # Return result
        return {
            "probability": score,
            "prediction": "Paired" if score > 0.5 else "Not Paired"
        }
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        app.logger.error(traceback.format_exc())
        return {"error": str(e)}

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error rendering index.html: {str(e)}")
        app.logger.error(traceback.format_exc())
        return f"Error: {str(e)}<br><br>Make sure you have a templates folder with index.html in it."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            app.logger.error("No JSON data received in request")
            return jsonify({"error": "No data received. Make sure you're sending JSON data."}), 400
            
        heavy_chain = data.get('heavy_chain', '')
        light_chain = data.get('light_chain', '')
        
        # Validate inputs
        if not heavy_chain or not light_chain:
            return jsonify({"error": "Both heavy and light chain sequences are required"}), 400
        
        # Make prediction
        result = predict_pairing(heavy_chain, light_chain)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in predict endpoint: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        data = request.json
        if not data:
            app.logger.error("No JSON data received in request")
            return jsonify({"success": False, "error": "No data received. Make sure you're sending JSON data."}), 400
            
        embedding_path = data.get('embedding_path', '')
        model_path = data.get('model_path', '')
        model_type = data.get('model_type', 'word2vec')
        k_mer = int(data.get('k_mer', 2))
        seq_max = int(data.get('seq_max', 9000))
        
        # Validate inputs
        if not embedding_path or not model_path:
            return jsonify({"success": False, "error": "Both embedding and model paths are required"}), 400
        
        # Check if files exist
        if not os.path.exists(embedding_path):
            return jsonify({"success": False, "error": f"Embedding file not found: {embedding_path}"}), 404
        
        if not os.path.exists(model_path):
            return jsonify({"success": False, "error": f"Model file not found: {model_path}"}), 404
        
        # Initialize models
        success = initialize_models(embedding_path, model_path, model_type, k_mer, seq_max)
        return jsonify({"success": success})
    except Exception as e:
        app.logger.error(f"Error in initialize endpoint: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/debug_info')
def debug_info():
    """Endpoint to get debug information"""
    try:
        # Get python path
        python_path = sys.path
        
        # Get current working directory
        cwd = os.getcwd()
        
        # Get directory contents
        dir_contents = os.listdir(cwd)
        
        # Check for templates directory
        templates_path = os.path.join(cwd, 'templates')
        templates_exists = os.path.exists(templates_path)
        
        templates_contents = []
        if templates_exists:
            templates_contents = os.listdir(templates_path)
        
        # Return debug info
        return jsonify({
            "python_path": python_path,
            "current_working_directory": cwd,
            "directory_contents": dir_contents,
            "templates_directory_exists": templates_exists,
            "templates_directory_contents": templates_contents,
            "device": str(device)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)