# Stride.AI-Task-1-Intent-detection-on-Enron-email-set
Task 1. Intent detection on Enron email set. We define "intent" here to correspond primarily to the categories "request" and "propose". In some cases, we also apply the positive label to some sentences from the "commit" category if they contain datetime, which makes them useful. Detecting the presence of intent in email is useful in many applications, e.g., machine mediation between human and email. The dataset contains parsed sentences from the email along with their intent (either 'yes' or 'no'). You need to build a learning model which detects whether a given sentence has intent or not. Its a 2-class classification problem. Although its not required you can refer this paper for more information on the dataset : Cohen, William W., Vitor R. Carvalho, and Tom M. Mitchell. "Learning to Classify Email into Speech Acts''." EMNLP. 2004.  Find the train and test dataset in enron.zip.  Tip: Try to add feature engineering into your model. Simple baseline with logistic regression gives 71% accuracy.  

Folder contains:
1. Train.txt
2. Test.txt
3. novel.py: A python script
4. Glove files

Run the script novel.py after running cd/<path_to_folder> 

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_5 (Embedding)      (None, 20, 300)           1746300   
_________________________________________________________________
dropout_10 (Dropout)         (None, 20, 300)           0         
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 16, 64)            96064     
_________________________________________________________________
dropout_11 (Dropout)         (None, 16, 64)            0         
_________________________________________________________________
conv1d_8 (Conv1D)            (None, 13, 32)            8224      
_________________________________________________________________
max_pooling1d_4 (MaxPooling1 (None, 3, 32)             0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 3, 32)             0         
_________________________________________________________________
lstm_4 (LSTM)                (None, 50)                16600     
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 51        
=================================================================
Total params: 1,867,239
Trainable params: 120,939
Non-trainable params: 1,746,300
_________________________________________________________________
None
Train on 2925 samples, validate on 732 samples
Epoch 1/4
2925/2925 [==============================] - 3s 1ms/step - loss: 0.6196 - acc: 0.6643 - val_loss: 0.8788 - val_acc: 0.1298
Epoch 2/4
2925/2925 [==============================] - 1s 452us/step - loss: 0.5466 - acc: 0.7265 - val_loss: 0.9730 - val_acc: 0.4044
Epoch 3/4
2925/2925 [==============================] - 1s 458us/step - loss: 0.4613 - acc: 0.7897 - val_loss: 0.9406 - val_acc: 0.4536
Epoch 4/4
2925/2925 [==============================] - 1s 464us/step - loss: 0.4069 - acc: 0.8260 - val_loss: 0.8746 - val_acc: 0.4658
Training Accuracy: 80.065628
Test Accuracy: 77.318548
