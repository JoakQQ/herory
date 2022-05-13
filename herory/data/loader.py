from herory.data import Tokenizer

class FileLoader():
    '''
        File loader for HERORY
        
        Arguments:
            - tokenizer (Tokenizer, optional): tokenizer to load dataset
            - path (string, default: `./datasets/merged_clean.txt`): path to dataset
            
        Functions:
            - load_exist_dict: load word dictionary from existing file
            - load_dict: load word dictionary from dataset
            - save_dict: save word dictionary to file
            - get_tokenizer: get tokenizer from file loader
    '''
    def __init__(self, tokenizer=None, path='./datasets/merged_clean.txt'):
        self.file_path = path
        self.tokenizer = tokenizer if tokenizer != None else Tokenizer()
        self.have_tokenizer = True if tokenizer != None else False

    def load_exist_dict(self, path='./datasets/dict.json'):
        '''
            Load word dictionary from existing file
            
            Input:
                path (string, default: `./datasets/dict.json`): input path of existing word dictionary
        '''
        self.tokenizer.load_exist_dict(path=path)

    def load_dict(self):
        '''
            Load word dictionary from dataset
        '''
        print(f'Reading file at {self.file_path} ...')
        with open(self.file_path, 'r') as f:
            for line in f:
                self.tokenizer.fit(line)
            f.close()
    
    def save_dict(self, path='./datasets/dict.json'):
        '''
            Save word dictionary to file
            
            Input:
                - path (string, default: `./datasets/dict.json`): output path of word dictionary
        '''
        self.tokenizer.save_dict(path=path)
        
    def get_tokenizer(self):
        '''
            Get tokenizer from file loader
            
            Output:
                - tokenizer (Tokenizer)
        '''
        return self.tokenizer