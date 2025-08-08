"""
This script provides utilities for creating and training an encoder-decoder model
"""
import torch
import torch.nn.functional as F
import optim
import tqdm
import numpy as np

from data.data_utils import sumTokenizer
from src.seq2seq.encoder_decoder import EncoderDecoder


def create_model(config):
    """
    Create a return a model instance. Using the configuration provided.
    :param config: model configuration
    :return: model: the configured model
    :return: device: the model is on
    :return: tokenizer: for data prep
    """
    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    tokenizer = sumTokenizer(config['max_text_length'], config['max_summary_length'])
    model = EncoderDecoder(
        vocab_size=tokenizer.vocab_size,
        emb_dim=config['emb_dim'],
        max_text_length=config['max_text_length'],
        max_summary_length=config['max_summary_length'],
        n_heads=config['text_heads'],
        n_layers=config['text_layers'],
    ).to(device)

    return model, device, tokenizer


def train_model(config, model, device, train_loader, smoke=False,
                val_loader=None, model_name='test'):
    """
    Train a large or small model.
    Right now this still prints to the screen.
    :param config: configuration settings
    :param model: model to train
    :param device: device the model is on
    :param train_loader: data loader for training set
    :param smoke: smoke test or large model
    :param val_loader: data loader for validation set (not for smoke)
    :param model_name: name for model file (not for smoke)
    :return: model: trained model
    """
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_history = []
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(config['epochs']):
        model.train()  # Ensure model is in training mode
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            src_mask = batch["input_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            tgt_mask = batch["target_mask"].to(device)

            if smoke:
                # smoke test is for tiny data
                # Forward pass with shifted decoder input
                outputs = model(
                    input_ids,
                    decoder_input_ids,  # shifted target IDs
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )

                # Compute loss (predicting `target_ids`)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1),
                    ignore_index=1
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Training loss:", loss.item())
                continue

            optimizer.zero_grad()
            # Mixed precision forward pass with shifted decoder input
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids,
                    decoder_input_ids,  # Use shifted targets as decoder input
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )

                # Calculate cross entropy loss (predicting `target_ids`)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),  # [B*summary_len, vocab_size]
                    target_ids.view(-1),  # [B*summary_len]
                    ignore_index=1  # Ignore padding tokens
                )

            loss_history.append(loss.item())

            # Backward pass with scaler for FP16 training
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Every 50 batches, print average loss and save model
            if (batch_idx + 1) % 50 == 0:
                avg_loss = np.mean(loss_history[-50:])
                progress_bar.set_postfix(loss=avg_loss)
                np.save("loss_history.npy", np.array(loss_history))
                torch.save(model.state_dict(), model_name + "_model.pt")

            # End-of-epoch: Evaluate on validation set
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    src_mask = batch["input_mask"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    decoder_input_ids = batch["decoder_input_ids"].to(device)
                    tgt_mask = batch["target_mask"].to(device)

                    outputs = model(
                        src=input_ids,
                        tgt=decoder_input_ids,  # Use shifted targets as decoder input
                        src_mask=src_mask,
                        tgt_mask=tgt_mask
                    )

                    loss = F.cross_entropy(
                        outputs.view(-1, outputs.size(-1)),
                        target_ids.view(-1),
                        ignore_index=1
                    )
                    val_losses.append(loss.item())

            avg_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch + 1}: Validation loss = {avg_val_loss}")

    return model
