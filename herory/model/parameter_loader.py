import json

def save_model_parameters(
    num_vocab=None, 
    hidden_size=None, 
    embedding_dim=None, 
    num_layers=None, 
    dropout=None, 
    teacher_force_ratio=None, 
    last_epoch=None,
    checkpoints_dir=None,
    path="./saves/model/parameters.json",
    ):
    """
        Save LSTM model parameters
        
        Inputs:
            - num_vocab (integer, default: `None`): number of vocabulary
            - hidden_size (integer, default: `None`): hidden size of LSTM
            - embedding_dim (integer, default: `None`): embedding dimensions of LSTM
            - num_layers (integer, default: `None`): number of LSTMs
            - dropout (float, default: `None`): dropout rate of LSTM
            - teacher_force_ratio (float, default: `None`): teacher force ratio
            - last_epoch (integer, default: `None`): last epoch the model trained on
            - checkpoints_dir (string, default: `None`): checkpoints directory
            - path (string, default: `./saves/model/parameters.json`): path to save the json file
    """
    save_json = {
        "num_vocab": num_vocab,
        "hidden_size": hidden_size,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "teacher_force_ratio": teacher_force_ratio,
        "last_epoch": last_epoch,
        "checkpoints_dir": checkpoints_dir
    }
    
    # Serializing json 
    json_object = json.dumps(save_json, indent = 4)

    # Writing to sample.json
    with open(path, "w") as outfile:
        outfile.write(json_object)
        outfile.close()
        
def get_model_parameters(path="./saves/model/parameters.json"):
    """
        Get model parameters from path
        
        Inputs:
            - path (string, default: `./saves/model/parameters.json`): path of the model parameters
    """
    json_object = None
    
    with open(path, 'r') as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        openfile.close()

    return json_object