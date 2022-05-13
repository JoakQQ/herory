import re
import json

class Tokenizer():
    '''
        HERORY Tokenizer
        
        Functions:
            - tokenize: white-space tokenize text into list
            - fit: fit text into tokenizer
            - load_exist_dict (call `FileLoader.load_exist_dict` instead): 
            load existing word dictionary into tokenizer
            - save_dict (call `FileLoader.save_dict` instead): save word dictionary
            - add_token: add customizerd token into tokenizer
            - get_word_to_ix: get word to index dictionary of tokenizer
            - text_to_sequence: get sequence from tokenizer with text input
            - sequence_to_text: get text from sequence
            - get_unknown_token_index: get unknown token index
            - get_start_token_index: get start of story token index
            - get_end_token_index: get end of story token index
            - get_padding_token_index: get padding token index
    '''
    def __init__(self):
        self.UNKNOWN = '<unk>'
        self.UNKNOWN_INDEX = 0
        self.START = '<start>'
        self.START_INDEX = 1
        self.END = '<end>'
        self.END_INDEX = 2
        self.PADDING = '<padding>'
        self.PADDING_INDEX = 3
        self.ix = 4
        self.word_to_ix = dict({self.UNKNOWN: self.UNKNOWN_INDEX, self.START: self.START_INDEX, self.END: self.END_INDEX, self.PADDING: self.PADDING_INDEX})
        self.ix_to_word = dict({self.UNKNOWN_INDEX: self.UNKNOWN, self.START_INDEX: self.START, self.END_INDEX: self.END, self.PADDING_INDEX: self.PADDING})
        self.patterns = [r'\'', r'\"', r'\.', r',', r'\(', r'\)', r'\!', r'\?', r'\;', r'\:', r'\_', r'\-\-']
        self.patterns_str = ["'", '"', '.', ',', '(', ')', '!', '?', ';', ':', '\\', '--']
        self.replacements = [' \' ', ' " ', ' . ', ' , ', ' ( ', ' ) ', ' ! ', ' ? ', ' ; ', ' : ', ' ', ' ',]
        self.patterns_dict = list((re.compile(p), r) for p, r in zip(self.patterns, self.replacements))

    def tokenize(self, line):
        '''
            Tokenize line
            
            Input:
                - line (string): texts to be tokenized
                
            Output:
                - tokenized words from line (list[string])
        '''
        line = line.lower()
        for pattern_re, replace in self.patterns_dict:
            line = pattern_re.sub(replace, line)
        return line.split()

    def fit(self, line):
        '''
            Fit line into tokenizer
            
            Input:
                - line (string): texts to be fit
        '''
        word_list = self.tokenize(line)
        for word in word_list:
            if word not in self.word_to_ix:
                self.word_to_ix[word] = self.ix
                self.ix_to_word[self.ix] = word
                self.ix += 1
    
    def load_exist_dict(self, path='./datasets/dict.json'):
        '''
            Load existing word dictionary into tokenizer
            
            (Not recommended to call this function from tokenizer, 
            call `FildLoader.load_exist_dict(path)` instead)
            
            Input:
                - path (string, default: `./datasets/dict.json`): file path to existing word dictionary
        '''
        print(f'Loading existing word dictionary from {path} ...')
        with open(path, 'r') as f:
            _json = json.load(f)
            self.word_to_ix = _json['word_to_ix']
            _ix_to_word = _json['ix_to_word']
            self.ix_to_word = dict()
            for _key, _value in _ix_to_word.items():
                self.ix_to_word[int(_key)] = _value
            self.ix = int(_json['ix'])
            f.close()
            
    def save_dict(self, path='./datasets/dict.json'):
        '''
            Save word dictionary to file
            
            (Not recommended to call this function from tokenizer, 
            call `FildLoader.save_dict(path)` instead)
            
            Input:
                - path (string, default: `./datasets/dict.json`): path of word dictionary
        '''
        print(f'Writing word dictionary to {path} ...')
        
        _json = {
            'word_to_ix': self.word_to_ix,
            'ix_to_word': self.ix_to_word,
            'ix': self.ix,
        }
        
        with open(path, "w") as f:
            json.dump(_json, f)
            f.close()
    
    def add_token(self, token):
        '''
            Add customized token to tokenizer
            
            Input:
                - token (string): customized token
        '''
        if token not in self.word_to_ix:
            self.word_to_ix[token] = self.ix
            self.ix_to_word[self.ix] = token
            self.ix += 1
    
    def get_word_to_ix(self, line=None):
        '''
            Get word to index for a line or the whole tokenizer
            
            Examples: 
            
                - `tokenizer.get_word_to_ix()` returns word to index for tokenizer
            
                - `tokenizer.get_word_to_ix("hello world")` returns `{'hello': 123, 'world': 90}`
            
            Input:
                - line (string, optional): input texts
        '''
        if line is None:
            return self.word_to_ix
        word_list = self.tokenize(line)
        word_to_ix = dict()
        for word in word_list:
            word_to_ix[word] = self.word_to_ix[word]
        return word_to_ix
    
    def get_unknown_token_index(self):
        '''
            Get unknown token index
            
            Output:
                - index of token (integer)
        '''
        return self.UNKNOWN_INDEX
    
    def get_start_token_index(self):
        '''
            Get start of story token index
            
            Output:
                - index of token (integer)
        '''
        return self.START_INDEX
    
    def get_end_token_index(self):
        '''
            Get end of story token index
            
            Output:
                - index of token (integer)
        '''
        return self.END_INDEX
    
    def get_padding_token_index(self):
        '''
            Get padding token index
            
            Output:
                - index of token (integer)
        '''
        return self.PADDING_INDEX
    
    def text_to_sequence(self, line):
        '''
            Turn texts into input sequence for model training purposes
            
            Input:
                - line (string): input texts
                
            Output:
                - sequence (list[integer])
        '''
        word_list = self.tokenize(line)
        text_seq = []
        for word in word_list:
            if word in self.word_to_ix:
                text_seq.append(self.word_to_ix[word])
            else:
                text_seq.append(self.UNKNOWN_INDEX)
        return text_seq
    
    def sequence_to_text(self, seq):
        '''
            Turn sequence into text
            
            Input:
                - seq (list[integer]): sequence
                
            Output:
                - text (string)
        '''
        text = ""
        for token in seq:
            word = self.ix_to_word[token]
            if text == "":
                text += word
            else:
                text += " " + word
        return text

    def sequence_to_text_sequence(self, seq):
        '''
            Turn sequence into text sequence
            
            Input:
                - seq (list[integer]): sequence
                
            Output:
                - text (list[string])
        '''
        text_seq = []
        for token in seq:
            text_seq.append(self.ix_to_word[token])
        return text_seq