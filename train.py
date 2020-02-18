from fastai.text import *
from pytorch_lamb import Lamb

vocabFilePath = 'vocab.txt'
trainFolderPath = ''
trainFileName = 'train.txt'

batchSize = 32
maxSequenceLength = 100
drop_mult = 0.1
qrnn = False
bidir = True

# Reading vocab file
with open(vocabFilePath, 'r') as file:
    tokens = file.read().replace('\n', '')
    
# Generate vocab class
bfdVocab = text.transform.Vocab.create(tokens, max_vocab=1000, min_freq=1)

# Print vocab size
print('Vocab size is ' + str(len(bfdVocab.itos)))

# Generate tokinizer
tok = Tokenizer(
    pre_rules = [fix_html, rm_useless_spaces],
    post_rules = [])
    
data_lm = TextLMDataBunch.from_csv(trainFolderPath, csv_name=trainFileName,text_cols=0,batchSize=32,vocab=bfdVocab,tokenizer=tok, include_eos=True,bptt=maxSequenceLength)

def modelConfig(qrnn:bool=False,bidir:bool=False):
    config = awd_lstm_lm_config.copy()
    config['emb_sz'] = 100
    config['n_hid'] = 4096
    config['n_layers'] = 4
    config['qrnn'] = qrnn
    config['bidir'] = bidir
    return config
    
learner = language_model_learner(data_lm, 
                               arch=AWD_LSTM, 
                               pretrained=False,
                               drop_mult=drop_mult,
                               config=config(qrnn=qrnn,bidir=bidir),
                               opt_func=Lamb)
                               
learn.fit_one_cycle(1, 1e-1)
