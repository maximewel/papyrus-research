import json


class IdCardCreator:

    @classmethod
    def create_lstm_id_card(cls) -> str:
        card = {}

        card[""] = 0

        return json.dumps(card, indent=4)
    
    @classmethod
    def create_transfo_id_card(cls, dataset_name, lr, epochs, batch_size,
                                    encoder_layers, decoder_layers, encoder_heads, decoder_heads,
                                    dropout_ratio, autoregressive_target_length, 
                                    patches_dim, embedding_dimension,
                                    use_prediction_token, use_lstm, lstm_model_path) -> str:
        card = {}

        card[TRAINING] = {
            DATASET: dataset_name,
            LR: lr,
            N_EPOCHS: epochs,
            BATCH_SIZE: batch_size,
            DROPOUT_RATIO: dropout_ratio,
        }

        card[STRUCTURAL] = {
            USE_PREDICTION_TOKEN: use_prediction_token,
            USE_LSTM: use_lstm,
            LSTM_MODEL_PATH: lstm_model_path,
            AUTOREGRESS_TARGET_LEN: autoregressive_target_length,
        }

        card[ENC_DEC] = {
            ENCODER_HEADS: encoder_heads,
            DECODER_HEADS: decoder_heads,
            ENCODER_LAYERS: encoder_layers,
            DECODER_LAYERS: decoder_layers,
        }

        card[DIMENSIONS] = {
            PATCHES_DIM: patches_dim,
            EMBEDDING_DIMS: embedding_dimension,
        }

        return json.dumps(card, indent=4)

TRAINING = "training"
DATASET = "dataset"
LR = "lr"
N_EPOCHS = "epochs"
BATCH_SIZE = "batch_size"
DROPOUT_RATIO = "dropout_ratio"

STRUCTURAL = "structural"
USE_PREDICTION_TOKEN = "use_prediction_token"
USE_LSTM = "use_lstm"
LSTM_MODEL_PATH = "lstm_model_path"
AUTOREGRESS_TARGET_LEN = "autoregressive_target_length"

ENC_DEC = "encoder_decorer"
ENCODER_HEADS = "encoder_heads"
DECODER_HEADS = "decoder_heads"
ENCODER_LAYERS = "encoder_layers"
DECODER_LAYERS = "decoder_layers"

DIMENSIONS = "dimensions"
PATCHES_DIM = "patches_dim"
EMBEDDING_DIMS = "embedding_dimension"


