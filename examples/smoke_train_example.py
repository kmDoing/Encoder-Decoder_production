"""
This script demonstrates a smoke training run of the encoder-decoder model
"""
import yaml
from src.seq2seq.model_utils import create_model, train_model
from src.seq2seq.eval_utils import test_model, plot_loss_history, score_model
import logging


data_ty = 'smoke'
from data.smoke_data import create_smoke_loaders
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(filename=data_ty+'_run.log', level=logging.INFO)
    base_path = '/Users/km_dh/PycharmProjects/encoder_decoder_prod'
    
    # Read in the config file
    try:
        with open(f'{base_path}/config/{data_ty}_config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Error: The file '{data_ty}_config.yaml' was not found.")
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML: {exc}")

    try:
        if config['emb_dim'] % config['n_heads'] != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
    except ValueError as e:
        logger.error(f"Invalid number of attention heads: {config['n_heads']}, error: {e}")
            
    # Get the data as a train_loader
    train_loader, val_loader, test_loader = create_smoke_loaders(config)

    model, device, tokenizer = create_model(config)

    model = train_model(config, model, device, train_loader, smoke=True,
                        val_loader=None, model_name='test')

    # Testing the model on a batch from the train loader (seen example)
    generated_output = test_model(train_loader, model, device, tokenizer,
                                  config['max_summary_length'])
    print(generated_output)

    # Testing the model on a batch from the test loader (unseen example)
    generated_output = test_model(test_loader, model, device, tokenizer,
                                  config['max_summary_length'])
    print(generated_output)

    # Check the loss is going down
    if 'smoke' not in data_ty:
        plot_loss_history(data_ty)
        scores = score_model(test_loader, model, device)
        print(scores)


if __name__ == "__main__":
    main()