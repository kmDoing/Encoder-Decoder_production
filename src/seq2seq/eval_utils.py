"""
This script provides utilities for examining, and testing the model
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer


def plot_loss_history(data_ty):
    """
    Plot the loss history of the training run
    :param: data_ty: the data type used to name the loss history file
    """
    # Load the loss history from the .npy file
    loss_history = np.load(f'{data_ty}_loss_history.npy')

    # Plot the loss history
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Training Loss Over Batches')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


def test_model(test_loader, model, device, tokenizer, max_summary_length):
    """
    Test the model on a batch from the specified data_loader
    :param: data_loader: test_loader with test data
    :param: model: model to test
    :param: device: device model is on
    :param: tokenizer: for tokenizing data
    :param: max_summary_length
    :return: string with generated output for the first test instance
    """
    # Testing the model on a batch from the train loader (seen example)
    test_batch = next(iter(test_loader))
    input_ids = test_batch["input_ids"].to(device)
    src_mask = test_batch["input_mask"].to(device)
    target_ids = test_batch["target_ids"].to(device)
    tgt_mask = test_batch["target_mask"].to(device)

    print("Test input:\n", tokenizer.tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print("\nTest target:", tokenizer.tokenizer.decode(target_ids[0], skip_special_tokens=True))

    model.eval()
    with torch.no_grad():
        outputs = model.generate(input_ids, src_mask=src_mask, max_length=max_summary_length)

    predicted_text = tokenizer.tokenizer.batch_decode(outputs, skip_special_tokens=True)  # Decode to text
    return f"Generated Output:{predicted_text[0]}"  # Print the predicted output


def score_model(test_loader, model, device):
    """
    Calculate the ROUGE-1, -2, and -L scores for the test data
    :param: test_loader: data loader for the test data
    :param: model: the model to test
    :param: device: the device (cpu/gpu) the model is on
    :return: dict of scores
    """
    output = {}
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    test_loss = 0
    all_references = []
    all_hypotheses = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    with torch.no_grad():  # Disable gradient calculation
        for batch in test_loader:
            # Move everything to device
            input_ids = batch["input_ids"].to(device)
            src_mask = batch["input_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            tgt_mask = batch["target_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids,
                target_ids,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )

            # Calculate cross entropy loss directly
            loss = F.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                target_ids.view(-1),
                ignore_index=1
            )

            test_loss += loss.item()

            # Convert outputs and targets to strings for ROUGE score
            predictions = outputs.argmax(dim=-1).tolist()
            references = target_ids.tolist()

            for pred, ref in zip(predictions, references):
                pred_str = ' '.join(map(str, pred))
                ref_str = ' '.join(map(str, ref))

                # Calculate ROUGE scores for each prediction and reference
                score = scorer.score(ref_str, pred_str)

                rouge_scores['rouge1'].append(score['rouge1'])
                rouge_scores['rouge2'].append(score['rouge2'])
                rouge_scores['rougeL'].append(score['rougeL'])

    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    output['test_loss'] = f"Test Loss: {avg_test_loss:.4f}"

    # Calculate average ROUGE scores
    avg_rouge1 = sum([score.fmeasure for score in rouge_scores['rouge1']]) / len(rouge_scores['rouge1'])
    avg_rouge2 = sum([score.fmeasure for score in rouge_scores['rouge2']]) / len(rouge_scores['rouge2'])
    avg_rougeL = sum([score.fmeasure for score in rouge_scores['rougeL']]) / len(rouge_scores['rougeL'])

    output['rouge_1'] = f"ROUGE-1 Score: {avg_rouge1:.4f}"
    output['rouge_2'] = f"ROUGE-2 Score: {avg_rouge2:.4f}"
    output['rouge_L'] = f"ROUGE-L Score: {avg_rougeL:.4f}"

    return output
