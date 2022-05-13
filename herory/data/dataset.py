import torch
from torch.utils.data import Dataset

from herory.data import Tokenizer

class StoryByStoryDataset(Dataset):
    '''
        Story by story dataset
           
        Every story is separated by 4 lines
        
        Arguments:
            - tokenizer (Tokenizer): tokenizer for the dataset
            - sequence_length (integer): sequence length for model
            - path (string, default: `./datasets/merged_clean.txt`): path to dataset
        
        Additional functions:
            - get_stories: get stories from dataset
            - get_num_vocab: get number of vocabulary
    '''
    def __init__(
        self, 
        tokenizer: Tokenizer, 
        sequence_length=128, 
        path='./datasets/merged_clean.txt'
        ):
        self.file_path = path
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
    
        self.stories = list()
        self.input_sequences = list()
        
        # initialize dataset
        print('Initializing dataset ...')
        self.init_dataset()
        
    def init_dataset(self):
        last_lines = ['', '', '', '']
        with open(self.file_path, 'r') as f:
            story = ''
            for line in f:
                if (last_lines[0] == '\n' 
                    and last_lines[1] == '\n' 
                    and last_lines[2] == '\n' 
                    and last_lines[3] == '\n' 
                    and line != '\n'
                    ):
                    self.stories.append(story)
                    story = ''
                story += line
                last_lines.pop(0)
                last_lines.append(line)
            if story != '':
                self.stories.append(story)
            f.close()
            
        for story in self.stories:
            self.input_sequences += \
                [self.tokenizer.get_start_token_index()] + \
                self.tokenizer.text_to_sequence(story) + \
                [self.tokenizer.get_end_token_index()]
            
    def get_stories(self):
        '''
            Get stories from dataset
            
            Output:
                - stories (list[string])
        '''
        return self.stories
    
    def get_num_vocab(self):
        '''
            Get number of vocabulary
            
            Output:
                - number of vocabulary (integer)
        '''
        return len(self.tokenizer.get_word_to_ix())
    
    def __len__(self):
        return len(self.input_sequences) - self.sequence_length
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.input_sequences[index : index + self.sequence_length]),
            torch.tensor(self.input_sequences[index + 1 : index + self.sequence_length + 1])
        )

class StoryByStoryMaskedDataset(Dataset):
    '''
        `*** Depreciated ***`
    
        Story by story dataset with `<unk>` token masking
           
        Every story is separated by 4 lines
        
        Arguments:
            - tokenizer (Tokenizer): tokenizer for the dataset
            - sequence_length (integer): sequence length for model
            - path (string, default: `./datasets/merged_clean.txt`): path to dataset
            - masking_prob (float, default: 0.1): masking probability of a line
        
        Additional functions:
            - get_stories: get stories from dataset
            - get_num_vocab: get number of vocabulary
    '''
    def __init__(
        self, 
        tokenizer: Tokenizer, 
        sequence_length=128, 
        path='./datasets/merged_clean.txt',
        masking_prob=0.1
        ):
        self.file_path = path
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.masking_prob = masking_prob
    
        self.stories = list()
        self.input_sequences = list()
        
        # initialize dataset
        print('Initializing dataset ...')
        self.init_dataset()
        
    def init_dataset(self):
        last_lines = ['', '', '', '']
        with open(self.file_path, 'r') as f:
            story = ''
            for line in f:
                if (last_lines[0] == '\n' 
                    and last_lines[1] == '\n' 
                    and last_lines[2] == '\n' 
                    and last_lines[3] == '\n' 
                    and line != '\n'
                    ):
                    self.stories.append(story)
                    story = ''
                story += line
                last_lines.pop(0)
                last_lines.append(line)
            if story != '':
                self.stories.append(story)
            f.close()
            
        for story in self.stories:
            self.input_sequences += \
                [self.tokenizer.get_start_token_index()] + \
                self.tokenizer.text_to_sequence(story, masking_prob=self.masking_prob) + \
                [self.tokenizer.get_end_token_index()]
            
    def get_stories(self):
        '''
            Get stories from dataset
            
            Output:
                - stories (list[string])
        '''
        return self.stories
    
    def get_num_vocab(self):
        '''
            Get number of vocabulary
            
            Output:
                - number of vocabulary (integer)
        '''
        return len(self.tokenizer.get_word_to_ix())
    
    def __len__(self):
        return len(self.input_sequences) - self.sequence_length
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.input_sequences[index : index + self.sequence_length]),
            torch.tensor(self.input_sequences[index + 1 : index + self.sequence_length + 1])
        )
