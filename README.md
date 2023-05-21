# Model Seq2Seq learning problems using Recurrent Neural Networks and Attention Networks
  Course Assignment of CS6910 Deep Learning IIT Madras
## Abstract<br/>
The Seq2Seq problem is built with ```Pytorch``` library, using Recurrent Neural Networks with different cells such 
as vanilla **RNN**, **LSTM** and **GRU**. The **Attention Network** is also built and compared with the vanilla model.
The notebooks are run on **Kaggle** using ```GPU P100```.
## Dataset<br/>
The experiment is performed with a sample of the ***Aksharantar dataset*** released by [AI4Bharat](https://ai4bharat.org/). This dataset contains pairs of the following form:
```xxx,yyy```, such as ```umanath,उमानाथ```. The model is trained on ***Hindi*** dataset ```(aksharantar_sampled/hin)```.
The dataset has training, validation, and test data already split into files ```hin_train.csv```, ```hin_valid.csv```, and ```hin_test.csv``` respectively. 
The training dataset has ***51,200*** examples whereas validation and test data has ***4096*** examples each.
## Objective<br/>
The goal is to ***transliterate*** a word ***(sequence of characters)*** given in the ***Latin*** script
to a word (sequence of characters) in the ***Devanagari*** script and compare the results obtained on using vanilla model and the model with attention networks.
The Seq2Seq models take longer time to train, so hyperparameter search was also efficiently performed on ```wandb```.
## Folder Structure<br/>
```
├── aksharantar_sampled
│   ├── **/*_train.csv
│   ├── **/*_test.csv
│   ├── **/*_valid.csv
├── predictions_attention
│   ├── pred_attn.csv
├── predictions_vanilla
│   ├── pred_vanilla.csv
├── script_predictions
│   ├── pred_script.csv
├── script_model
│   ├── best_model.pth
├── best_model.pth
├── best_model_attn.pth
├── Q2.ipynb
├── Q4.ipynb
├── Q4_c.ipynb
├── Q5.ipynb
├── Q5_a.ipynb
├── Q5_c.ipynb
├── script.py
├── script_attn.py
```
