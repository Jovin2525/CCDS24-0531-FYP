import numpy as np
import time
import torch
import os

def train(
    data,
    model,
    save_progress=False,
    save_parameters=False,
    num_updates=300000,
    verbose=True,
    job_string="",
    embeddings=False,
    update_offset=0,
    print_neff=True,
    print_iter=100
):
    """
    Main function to train DeepSequence

    Parameters
    --------------
    data: Instance of DataHelper class from helper.py
    model: Instance of VariationalAutoencoder or similar PyTorch-based model
    save_progress: Save log files of losses during training
    save_parameters: Save parameters every k iterations
    num_updates: Number of training iterations (int)
    verbose: Print output during training
    job_string: String by which to save all summary files during training
    embeddings: Save latent variables every k iterations (int)
                or "log": Save latent variables during training on log scale iterations
                or False (bool)
    update_offset: Offset to use for training
    print_neff: Print the Neff of the alignment
    print_iter: Print/write out losses every k iterations

    Returns
    ------------
    None
    """

    batch_size = model.batch_size
    batch_order = np.arange(data.x_train.shape[0])

    seq_sample_probs = data.weights / np.sum(data.weights)

    update_num = 0

    # Initialize the optimizer with the given offset
    if update_offset != 0:
        model.optimizer.param_groups[0]['initial_lr'] += update_offset

    LB_list = []
    loss_params_list = []
    KLD_latent_list = []
    reconstruct_list = []

    if save_progress:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(data.working_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        err_filename = f"{data.working_dir}/logs/{job_string}_err.csv"
        with open(err_filename, "w") as output:
            if print_neff:
                output.write(f"Neff:\t{data.Neff}\n")

    start_time = time.time()

    # Calculate the indices to save latent variables
    if embeddings == "log":
        start_embeddings = 10
        log_embedding_interpolants = sorted(
            set(
                np.floor(
                    np.exp(np.linspace(np.log(start_embeddings), np.log(50000), 1250))
                ).tolist()
            )
        )
        log_embedding_interpolants = [int(val) for val in log_embedding_interpolants]

    while (update_num + update_offset) < num_updates:
        update_num += 1

        batch_index = np.random.choice(
            batch_order, batch_size, p=seq_sample_probs
        ).tolist()

        x_batch = torch.tensor(data.x_train[batch_index], dtype=torch.float32)

        # Forward and backward pass with the model
        batch_LB, batch_reconstr_entropy, batch_loss_params, batch_KLD_latent = model.update(
            x_batch, data.Neff, update_num
        )

        LB_list.append(batch_LB)
        loss_params_list.append(batch_loss_params)
        KLD_latent_list.append(batch_KLD_latent)
        reconstruct_list.append(batch_reconstr_entropy)

        if save_parameters and update_num % save_parameters == 0:
            if verbose:
                print("Saving Parameters")
            # Create params directory if does not exist
            params_dir = os.path.join(data.working_dir, "params")
            os.makedirs(params_dir, exist_ok=True)
            model.save_parameters(f"{job_string}_epoch-{update_num+update_offset}")

        # Save embeddings at log intervals or regular intervals
        if embeddings:
            if embeddings == "log":
                if update_num + update_offset in log_embedding_interpolants:
                    # Create embeddings directory if does not exist
                    embeddings_dir = os.path.join(data.working_dir, "embeddings")
                    os.makedirs(embeddings_dir, exist_ok=True)
                    data.get_embeddings(model, update_num + update_offset, filename_prefix=job_string)
            elif update_num % embeddings == 0:
                # Create embeddings directory if does not exist
                embeddings_dir = os.path.join(data.working_dir, "embeddings")
                os.makedirs(embeddings_dir, exist_ok=True)
                data.get_embeddings(model, update_num + update_offset, filename_prefix=job_string)

        if update_num % print_iter == 0:
            mean_index = np.arange(update_num - print_iter, update_num)

            LB = np.mean(np.asarray(LB_list)[mean_index])
            KLD_params = np.mean(np.asarray(loss_params_list)[mean_index])
            KLD_latent = np.mean(np.asarray(KLD_latent_list)[mean_index])
            reconstruct = np.mean(np.asarray(reconstruct_list)[mean_index])

            progress_string = (
                f"Update {update_num + update_offset} finished. LB : {LB:.2f},  Params: {KLD_params:.2f}, "
                f"Latent: {KLD_latent:.2f}, Reconstruct: {reconstruct:.2f}, Time: {time.time() - start_time:.2f}"
            )

            start_time = time.time()

            if verbose:
                print(progress_string)

            if save_progress:
                with open(err_filename, "a") as output:
                    output.write(progress_string + "\n")