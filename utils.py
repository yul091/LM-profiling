import time
import torch



class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        
   
# Function to get current GPU memory usage
def get_memory():
    return torch.cuda.memory_allocated()


# def extract_seq_cls_model_layers(
#     model: Union[
#         BertForSequenceClassification, 
#         GPT2ForSequenceClassification,  
#         BartForSequenceClassification,
#     ]
# ):
#     model_name = model.__class__.__name__.lower()
#     layers = []
#     if "bert" in model_name:
#         layers = model.bert.encoder.layer
#     elif "gpt2" in model_name:
#         layers = model.transformer.h
#     elif "bart" in model_name:
#         encoder_layers = model.model.encoder.layers
#         decoder_layers = model.model.decoder.layers
#         layers = encoder_layers + decoder_layers
#     return layers