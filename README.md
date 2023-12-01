# POS and NER Tagging via BiLSTM + CNN

Usage: 

```
main.py [-h] [--task TASK [TASK ...]] [--train] [--predict] [--eval] [--seed SEED] [--batch BATCH] [--epoch EPOCH] [--lr LR] [--momentum MOMENTUM] [--word-length WORD_LENGTH]

options:
  -h, --help            show this help message and exit
  --task TASK [TASK ...]
                        Select task: 1 for task 1, 2 for task 2, 3 for bonus task (default [1, 2, 3])
  --train               Train and save a new model base on selected hyper-parameters on selected task(s)
  --predict             Output the prediction on dev set and test set into files(default False)
  --eval                Evaluate model(s) on dev set using Perl script "conll03eval.txt" (default False)
  --seed SEED           Set random seed (default 0)
  --batch BATCH         Set batch size (default 256)
  --epoch EPOCH         Set epoch number (default 100)
  --lr LR               Set learning rate (default 0.5)
  --momentum MOMENTUM   Set momentum (default 0.9)
  --word-length WORD_LENGTH
                        Set total length of word for character embedding
```



e.g.1:
python main.py --train --predict --eval
for training all tasks, output the prediction into files, and evaluate by the Perl script conll03eval.txt

e.g.2:
python main.py --task 1 2 --predict
for loading model from blstm1.out and blstm2.out for task 1 and 2 and output the prediction into files dev1.out, test1.out, dev2.out, test2.out