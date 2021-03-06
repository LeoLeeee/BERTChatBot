# BERTChatBot

Using Transformer and BERT to code a Chat Bot

## Files' function

1. seq2seq.ini and getConfig.py are used to config and load the parameter used in the project, such as the path of the checkpoint.
2. plot_util.py and Utility.py are copy the function in "dive into deep learning" to plot data and get some convenient tools.
3. AttentionModel.py and BERTModel.py are copy and revised based on the "dive into deep learning" to create Transformer and BERT modules.
4. data_tokenize.ipynb is used to demonstrate the loading and tokenize of the data from the "train_data/xiaohuangji50w_nofenci.conv". The data_tokenize.py is used to save the useful function for other scripts to import.
5. data_load_for_Transformer/Bert.py/ipynb are used to construct train_iter for the training from the "train_data/xiaohuangji50w_nofenci.conv" based on the tokenized vocabulary.
6. BERT_pretraining_single.py/ipynb are used to pretraining the Bert model and save the checkpoint in the "model_data/model_BERT_pretraing_single.pt".
7. excute.py/ipynb are used to train the chat bot net with transformer encoder and decoder, excute_bert.py/ipynb are the net with bert encoder, excute_bert_ed.py/ipynb are the net with bert encoder and decoder.
8. app.py builds a web to chat with bot based on JS.

## How to use the code

1. Download the code on your computer.
2. You may need to create a file named "model_data" to save the checkpoint.
3. Run the excute.ipynb until the train loss converge. This may take several tens hours based on the 1060Ti GPU.
4. Importing excute.py file in the app.py to use the trained model.
5. Pretraining the BERT model with the BERT_pretraining_single.ipynb before the excute_bert.ipynb or excute_bert_ed.ipynb to train the BERT model.
6. Importing excute_bert.py or excute_bert_ed.py file in the app.py to use the trained model.

## Note

1. I find the pretrained model is not act better obviously than the one use only Transformer encoder. It may because the data is not well?
2. I tried to fix the bert parameters or not pretrained by the pretraining_Bert_single.ipynb during training the whole model, the trained loss is better with not fixing the bert model parameters.