import random
import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from happytransformer import GENSettings

def bleu_gpt(model, tokenizer, dataloader, prob=0.1):
    """
        BLEU score for GPT-Neo model
        
        inputs:
            - model: GPT-Neo model
            - tokenizer: HERORY Tokenizer
            - dataloader: Data loader
            - prob (default: `0.1`): ratio of sample in dataset used for calculating BLEU
    """
    
    # text generation settings
    generic_sampling_settings = GENSettings(do_sample=True, top_k=0, temperature=0.9,  max_length=1)

    total_bleu = []
    
    for _, (x, y) in enumerate(dataloader):
        if random.random() > prob: 
            continue
        
        batch = random.randint(0, x.shape[0] - 1)
        
        input_text = tokenizer.sequence_to_text(x[batch].tolist())
        
        output_generic_sampling = model.generate_text(input_text, args=generic_sampling_settings)
        
        output = input_text + output_generic_sampling.text
    
        ref = tokenizer.tokenize(input_text)[-16:] + tokenizer.sequence_to_text_sequence([int(y[batch][-1])])
        hypo = tokenizer.tokenize(output)[-17]
        
        score_bleu = sentence_bleu(
            [ref], 
            hypo, 
            weights=[1]
            )

        total_bleu.append(score_bleu)

    return sum(total_bleu)/len(total_bleu)
    

def bleu_lstm(model, tokenizer, device, dataloader, prob=0.1):
    """
        BLEU score for HERORY LSTM model
        
        inputs:
            - model: HERORY LSTM model
            - tokenizer: HERORY Tokenizer
            - device: device the model is on
            - dataloader: Data loader
            - prob (default: `0.1`): ratio of sample in dataset used for calculating BLEU
    """
    
    model.eval()
    model.to(device)
    
    total_bleu = []
    
    for _, (x, y) in enumerate(dataloader):
        if random.random() > prob: 
            continue
        
        batch = random.randint(0, x.shape[0] - 1)
        
        input_text = tokenizer.sequence_to_text(x[batch].tolist())
        input_seq = tokenizer.text_to_sequence(input_text)
        
        state_h, state_c = model.init_state(len(input_seq))

        x = torch.tensor([input_seq]).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        
        last_word_logits = y_pred[0][-1]
        p = F.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
    
        ref = tokenizer.sequence_to_text_sequence(input_seq[-16:]) + tokenizer.sequence_to_text_sequence([int(y[batch][-1])])
        hypo = tokenizer.sequence_to_text_sequence(input_seq[-16:]) + tokenizer.sequence_to_text_sequence([word_index])
        
        score_bleu = sentence_bleu([ref], hypo, weights=[1])
        
        total_bleu.append(score_bleu)

    return sum(total_bleu)/len(total_bleu)