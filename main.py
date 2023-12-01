import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset, TensorDataset
import gzip
import argparse
from tqdm import tqdm, trange


class BLSTM(nn.Module):
    def __init__(self, num_words, num_tags=9, emb_dim=100, hidden_dim=256, dropout=0.33, linear_dim=128,
                 emb_weights=None):
        super(BLSTM, self).__init__()
        self.num_words = num_words
        if emb_weights is None:
            self.embedding = nn.Embedding(num_words, emb_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(emb_weights, padding_idx=0, freeze=False)
        self.dropout = nn.Dropout(p=dropout)
        self.blstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(linear_dim, num_tags)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.blstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=0)
        out = self.dropout(out)
        out = self.elu(self.fc1(out))
        out = self.fc2(out)
        return out


class CNNBLSTM(nn.Module):
    def __init__(self, num_words, num_tags=9, emb_dim=100, hidden_dim=256, dropout=0.33, linear_dim=128,
                 emb_weights=None, char_vocab_size=55, char_rep_dim=1, cnn_emb_dim=30, word_length=50,
                 conv_out_dim=3, kernel_size=3, maxpool_size=5):
        super(CNNBLSTM, self).__init__()
        self.num_words = num_words
        self.char_rep_dim = char_rep_dim
        self.word_length = word_length
        self.char_embedding = nn.Embedding(char_vocab_size, char_rep_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(char_rep_dim, conv_out_dim, kernel_size, padding=1)
        self.maxpool = nn.MaxPool1d(maxpool_size, ceil_mode=True)
        if emb_weights is None:
            self.word_embedding = nn.Embedding(num_words, emb_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(emb_weights, padding_idx=0, freeze=False)
        self.dropout = nn.Dropout(p=dropout)
        self.blstm = nn.LSTM(emb_dim + cnn_emb_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim*2, linear_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(linear_dim, num_tags)

    def forward(self, x, lengths, characters):
        x = self.word_embedding(x)
        characters = self.char_embedding(characters).reshape(-1, self.char_rep_dim, self.word_length)
        characters = self.conv1d(characters)
        characters = self.maxpool(characters).reshape(x.size(0), x.size(1), -1)
        x = torch.cat([x, characters], dim=2)
        packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.blstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=0)
        out = self.dropout(out)
        out = self.elu(self.fc1(out))
        out = self.fc2(out)
        return out


class MyTrainData(Dataset):
    def __init__(self, data, character, label):
        self.data = data
        self.character = character
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.character[item], self.label[item]


class MyTestData(Dataset):
    def __init__(self, data, character):
        self.data = data
        self.character = character

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.character[item]


def train_collate_fn(dataset):
    dataset = sorted(dataset, key=lambda sample: len(sample[0]), reverse=True)
    data, character, label = zip(*dataset)
    data_length = [len(sample) for sample in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    character = rnn_utils.pad_sequence(character, batch_first=True, padding_value=0)
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=-100)
    return data, character, label, data_length


def test_collate_fn(dataset):
    data, character = zip(*dataset)
    data_length = [len(sample) for sample in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    character = rnn_utils.pad_sequence(character, batch_first=True, padding_value=0)
    return data, character, data_length


def get_glove_data(train_sentence_table, dev_sentence_table, test_sentence_table, batch_size,
                   train_label_tensor_table, train_word_tensor_table,
                   dev_word_tensor_table, dev_label_tensor_table, test_word_tensor_table):
    glove_idx = 3
    glove_vocab = dict()
    glove_weight = [torch.zeros(100), torch.zeros(100), torch.zeros(100)]
    with gzip.open('glove.6B.100d.gz', 'r') as fp:
        for line in fp:
            line_list = line.split()
            word = line_list[0].decode('utf8')
            vec = [float(e) for e in line_list[1:]]
            glove_weight.append(torch.tensor(vec))
            glove_weight.append(torch.tensor(vec))
            glove_vocab[word] = glove_idx
            glove_idx += 2
    glove_weight = torch.stack(glove_weight)

    glove_train_data_tensor_table = []
    for sentence in train_sentence_table:
        glove_train_data_tensor_table.append(torch.tensor([get_glove_idx(glove_vocab, token[1]) for token in sentence]))

    glove_dev_data_tensor_table = []
    for sentence in dev_sentence_table:
        glove_dev_data_tensor_table.append(torch.tensor([get_glove_idx(glove_vocab, token[1]) for token in sentence]))

    glove_test_data_tensor_table = []
    for sentence in test_sentence_table:
        glove_test_data_tensor_table.append(torch.tensor([get_glove_idx(glove_vocab, token[1]) for token in sentence]))

    glove_train_dataloader, glove_eval_dataloader, glove_dev_dataloader, glove_test_dataloader \
        = get_dataloader(batch_size, glove_train_data_tensor_table, train_word_tensor_table, train_label_tensor_table,
                         glove_dev_data_tensor_table, dev_word_tensor_table, dev_label_tensor_table,
                         glove_test_data_tensor_table, test_word_tensor_table)

    return glove_idx, glove_weight, \
           glove_train_dataloader, glove_eval_dataloader, glove_dev_dataloader, glove_test_dataloader


def get_dataloader(batch_size, train_data_tensor_table, train_word_tensor_table, train_label_tensor_table,
                   dev_data_tensor_table, dev_word_tensor_table, dev_label_tensor_table,
                   test_data_tensor_table, test_word_tensor_table):
    train_data = MyTrainData(train_data_tensor_table, train_word_tensor_table, train_label_tensor_table)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=train_collate_fn)
    eval_data = MyTrainData(dev_data_tensor_table, dev_word_tensor_table, dev_label_tensor_table)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, collate_fn=train_collate_fn)
    dev_data = MyTestData(dev_data_tensor_table, dev_word_tensor_table)
    dev_dataloader = DataLoader(dev_data, batch_size=batch_size, collate_fn=test_collate_fn)
    test_data = MyTestData(test_data_tensor_table, test_word_tensor_table)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=test_collate_fn)
    return train_dataloader, eval_dataloader, dev_dataloader, test_dataloader


def get_tensor_tables(word_length, word_vocab, tag_vocab,
                      train_sentence_table, dev_sentence_table, test_sentence_table):
    train_data_tensor_table = []
    train_word_tensor_table = []
    train_label_tensor_table = []
    for sentence in train_sentence_table:
        train_data_tensor_table.append(torch.tensor([get_vocab_idx(word_vocab, token[1]) for token in sentence]))
        train_word_tensor_table.append(torch.tensor([get_char_seq(token[1], word_length) for token in sentence]))
        train_label_tensor_table.append(torch.tensor([tag_vocab[token[2]] for token in sentence]))
    dev_data_tensor_table = []
    dev_word_tensor_table = []
    dev_label_tensor_table = []
    for sentence in dev_sentence_table:
        dev_data_tensor_table.append(torch.tensor([get_vocab_idx(word_vocab, token[1]) for token in sentence]))
        dev_word_tensor_table.append(torch.tensor([get_char_seq(token[1], word_length) for token in sentence]))
        dev_label_tensor_table.append(torch.tensor([tag_vocab[token[2]] for token in sentence]))
    test_data_tensor_table = []
    test_word_tensor_table = []
    for sentence in test_sentence_table:
        test_data_tensor_table.append(torch.tensor([get_vocab_idx(word_vocab, token[1]) for token in sentence]))
        test_word_tensor_table.append(torch.tensor([get_char_seq(token[1], word_length) for token in sentence]))
    return train_data_tensor_table, train_word_tensor_table, train_label_tensor_table, dev_data_tensor_table, \
           dev_word_tensor_table, dev_label_tensor_table, test_data_tensor_table, test_word_tensor_table


def get_vocab_idx(vocab, word):
    if word in vocab:
        return vocab[word]
    elif word[0].isupper():
        return 2
    else:
        return 1


def get_glove_idx(vocab, word):
    if word.lower() in vocab:
        idx = vocab[word.lower()]
    else:
        idx = 1
    if word[0].isupper():
        return idx + 1
    else:
        return idx


def get_char_num(character):
    if character.isupper():
        return ord(character) - ord('A') + 1
    if character.islower():
        return ord(character) - ord('a') + 27
    if character.isnumeric():
        return 53
    return 54


def get_char_seq(word, word_len):
    return [get_char_num(word[idx]) if idx < len(word) else 0 for idx in range(word_len)]


def read_file():
    fp = open('data/train', 'r')
    sentence = []
    train_sentence_table = []
    word_counter = dict()
    tag_set = set()
    threshold = 2
    for line in fp:
        if line != '\n':
            line_list = line.split()
            sentence.append(line_list)
            if line_list[1] not in word_counter:
                word_counter[line_list[1]] = 1
            else:
                word_counter[line_list[1]] += 1
            tag_set.add(line_list[2])
        else:
            train_sentence_table.append(sentence)
            sentence = []
    if sentence:
        train_sentence_table.append(sentence)
    fp.close()
    common_words = [key for key in word_counter if word_counter[key] >= threshold]
    word_vocab = {word: i + 3 for i, word in enumerate(common_words)}
    word_vocab['<unk>'] = 1
    word_vocab['<UNK>'] = 2
    tag_vocab = {tag: i for i, tag in enumerate(sorted(tag_set, reverse=True))}
    tag_inverse = {v: k for k, v in tag_vocab.items()}

    fp = open('data/dev', 'r')
    sentence = []
    dev_sentence_table = []
    for line in fp:
        if line != '\n':
            line_list = line.split()
            sentence.append(line_list)
        else:
            dev_sentence_table.append(sentence)
            sentence = []
    if sentence:
        dev_sentence_table.append(sentence)
    fp.close()

    fp = open('data/test', 'r')
    sentence = []
    test_sentence_table = []
    for line in fp:
        if line != '\n':
            line_list = line.split()
            sentence.append(line_list)
        else:
            test_sentence_table.append(sentence)
            sentence = []
    if sentence:
        test_sentence_table.append(sentence)
    fp.close()

    return word_vocab, tag_vocab, tag_inverse, train_sentence_table, dev_sentence_table, test_sentence_table


def train_model(model, device, criterion, optimizer, train_dataloader, cnn=False):
    model.train()
    avg_loss = 0.
    for i, (data, character, label, length) in enumerate(train_dataloader):
        data, character, label = data.to(device), character.to(device), label.to(device)
        if cnn:
            output = model(data, length, character).reshape(-1, 9)
        else:
            output = model(data, length).reshape(-1, 9)
        label = label.reshape(-1)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

    avg_loss /= len(train_dataloader)
    return avg_loss


def evaluate_model(model, device, test_dataloader, cnn=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, character, label, length) in enumerate(test_dataloader):
            data, character, label = data.to(device), character.to(device), label.to(device)
            if cnn:
                output = model(data, length, character)
            else:
                output = model(data, length)
            mask = label != -100
            label = label[mask]
            predict_label = output.argmax(dim=2)[mask]
            correct += (predict_label == label).sum().item()
            total += mask.sum().item()
    return correct / total


def predict(model, device, test_loader, cnn=False):
    model.eval()
    prediction = []
    with torch.no_grad():
        for data, character, lengths in test_loader:
            data, character = data.to(device), character.to(device)
            if cnn:
                outputs = model(data, lengths, character)
            else:
                outputs = model(data, lengths)
            for out, l in zip(outputs, lengths):
                prediction.extend(torch.argmax(out, dim=1)[:l].tolist())
    return prediction


def output_prediction(in_filename, out_filename, prediction, tag_inverse, evaluate=False):
    fin = open(in_filename, 'r')
    fout = open(out_filename, 'w')
    line_counter = 0
    for line in fin:
        line_list = line.split()
        if line != '\n':
            if evaluate:
                fout.write(line[:-1] + ' ' + tag_inverse[prediction[line_counter]] + '\n')
            else:
                fout.write(line_list[0] + ' ' + line_list[1] + ' ' + tag_inverse[prediction[line_counter]] + '\n')
            line_counter += 1
        else:
            fout.write(line)


def main(main_params):
    torch.manual_seed(main_params.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = main_params.batch
    lr = main_params.lr
    momentum = main_params.momentum

    word_vocab, tag_vocab, tag_inverse, train_sentence_table, dev_sentence_table, test_sentence_table = read_file()
    train_data_tensor_table, train_word_tensor_table, train_label_tensor_table, dev_data_tensor_table,\
    dev_word_tensor_table, dev_label_tensor_table, test_data_tensor_table, test_word_tensor_table\
        = get_tensor_tables(main_params.word_length, word_vocab, tag_vocab,
                            train_sentence_table, dev_sentence_table, test_sentence_table)
    train_dataloader, eval_dataloader, dev_dataloader, test_dataloader \
        = get_dataloader(batch_size, train_data_tensor_table, train_word_tensor_table, train_label_tensor_table,
                         dev_data_tensor_table, dev_word_tensor_table, dev_label_tensor_table,
                         test_data_tensor_table, test_word_tensor_table)

    if 2 in main_params.task or 3 in main_params.task:
        glove_size, glove_weight, glove_train_dataloader, glove_eval_dataloader, \
        glove_dev_dataloader, glove_test_dataloader = \
            get_glove_data(train_sentence_table, dev_sentence_table, test_sentence_table,
                           batch_size, train_label_tensor_table, train_word_tensor_table,
                           dev_word_tensor_table, dev_label_tensor_table, test_word_tensor_table)

    if main_params.train:
        if 1 in main_params.task:
            print("Training model for task 1")
            torch.manual_seed(main_params.seed)
            blstm = BLSTM(num_words=len(word_vocab) + 1)
            blstm.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(blstm.parameters(), lr=lr, momentum=momentum)
            pbar = trange(main_params.epoch, unit='epoch', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            best_eval_acc = 0
            best_epoch = 0
            for epoch in pbar:
                avg_loss = train_model(blstm, device, criterion, optimizer, train_dataloader)
                eval_acc = evaluate_model(blstm, device, eval_dataloader)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_epoch = epoch
                    torch.save(blstm.state_dict(), 'blstm1.pt')
                pbar.set_postfix({"training loss": avg_loss, "validation accuracy": eval_acc, "best":best_eval_acc})
            print("Task 1 training finished, best validation accuracy appears at epoch %i" % best_epoch)

        if 2 in main_params.task:
            print("Training model for task 2")
            torch.manual_seed(main_params.seed)
            glove_blstm = BLSTM(num_words=glove_size, emb_weights=glove_weight)
            glove_blstm.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(glove_blstm.parameters(), lr=lr, momentum=momentum)
            pbar = trange(main_params.epoch, unit='epoch', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            best_eval_acc = 0
            best_epoch = 0
            for epoch in pbar:
                avg_loss = train_model(glove_blstm, device, criterion, optimizer, glove_train_dataloader)
                eval_acc = evaluate_model(glove_blstm, device, glove_eval_dataloader)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_epoch = epoch
                    torch.save(glove_blstm.state_dict(), 'blstm2.pt')
                pbar.set_postfix({"training loss": avg_loss, "validation accuracy": eval_acc, "best":best_eval_acc})
            print("Task 2 training finished, best validation accuracy appears at epoch %i" % best_epoch)

        if 3 in main_params.task:
            print("Training model for bonus task")
            torch.manual_seed(main_params.seed)
            cnn_blstm = CNNBLSTM(num_words=glove_size, emb_weights=glove_weight)
            cnn_blstm.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(cnn_blstm.parameters(), lr=lr, momentum=momentum)
            pbar = trange(main_params.epoch, unit='epoch', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
            best_eval_acc = 0
            best_epoch = 0
            for epoch in pbar:
                avg_loss = train_model(cnn_blstm, device, criterion, optimizer, glove_train_dataloader, cnn=True)
                eval_acc = evaluate_model(cnn_blstm, device, glove_eval_dataloader, cnn=True)
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_epoch = epoch
                    torch.save(cnn_blstm.state_dict(), 'cnn-blstm.pt')
                pbar.set_postfix({"training loss": avg_loss, "validation accuracy": eval_acc, "best":best_eval_acc})
            print("Bonus task training finished, best validation accuracy appears at epoch %i" % best_epoch)

    if 1 in main_params.task:
        blstm = BLSTM(num_words=len(word_vocab) + 1)
        blstm.to(device)
        blstm.load_state_dict(torch.load('blstm1.pt'))
    if 2 in main_params.task:
        glove_blstm = BLSTM(num_words=glove_size, emb_weights=glove_weight)
        glove_blstm.to(device)
        glove_blstm.load_state_dict(torch.load('blstm2.pt'))
    if 3 in main_params.task:
        cnn_blstm = CNNBLSTM(num_words=glove_size, emb_weights=glove_weight)
        cnn_blstm.to(device)
        cnn_blstm.load_state_dict(torch.load('cnn-blstm.pt'))

    if main_params.predict:
        if 1 in main_params.task:
            dev_pred = predict(blstm, device, dev_dataloader)
            output_prediction('data/dev', 'dev1.out', dev_pred, tag_inverse)
            test_pred = predict(blstm, device, test_dataloader)
            output_prediction('data/test', 'test1.out', test_pred, tag_inverse)
            print("Prediction on dev and test set of task 1 in dev1.out and test1.out")
        if 2 in main_params.task:
            dev_pred = predict(glove_blstm, device, glove_dev_dataloader)
            output_prediction('data/dev', 'dev2.out', dev_pred, tag_inverse)
            test_pred = predict(glove_blstm, device, glove_test_dataloader)
            output_prediction('data/test', 'test2.out', test_pred, tag_inverse)
            print("Prediction on dev and test set of task 2 in dev2.out and test2.out")
        if 3 in main_params.task:
            test_pred = predict(cnn_blstm, device, glove_test_dataloader, cnn=True)
            output_prediction('data/test', 'pred', test_pred, tag_inverse)
            print("Prediction on test set of bonus task in pred")

    if main_params.eval:
        if 1 in main_params.task:
            print("Evaluating task 1")
            dev_pred = predict(blstm, device, dev_dataloader)
            output_prediction('data/dev', 'eval1', dev_pred, tag_inverse, evaluate=True)
            os.system("Perl conll03eval.txt < eval1")
        if 2 in main_params.task:
            print("Evaluating task 2")
            dev_pred = predict(glove_blstm, device, glove_dev_dataloader)
            output_prediction('data/dev', 'eval2', dev_pred, tag_inverse, evaluate=True)
            os.system("Perl conll03eval.txt < eval2")
        if 3 in main_params.task:
            print("Evaluating bonus task")
            dev_pred = predict(cnn_blstm, device, glove_dev_dataloader, cnn=True)
            output_prediction('data/dev', 'eval3', dev_pred, tag_inverse, evaluate=True)
            os.system("Perl conll03eval.txt < eval3")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=int, nargs="+", default=[1, 2, 3],
                        help="Select task: 1 for task 1, 2 for task 2, 3 for bonus task (default [1, 2, 3])")
    parser.add_argument('--train', action="store_true", default=False,
                        help="Train and save a new model base on selected hyper-parameters on selected task(s)")
    parser.add_argument('--predict', action="store_true", default=False,
                        help="Output the prediction on dev set and test set into files(default False)")
    parser.add_argument('--eval', action="store_true", default=False,
                        help="Evaluate model(s) on dev set using Perl script \"conll03eval.txt\" (default False)")
    parser.add_argument('--seed', default=0, help="Set random seed (default 0)")
    parser.add_argument('--batch', type=int, default=256, help="Set batch size (default 256)")
    parser.add_argument('--epoch', type=int, default=100, help="Set epoch number (default 100)")
    parser.add_argument('--lr', type=float, default=0.5, help="Set learning rate (default 0.5)")
    parser.add_argument('--momentum', type=float, default=0.9, help="Set momentum (default 0.9)")
    parser.add_argument('--word-length', type=int, default=50, help="Set total length of word for character embedding")

    args = parser.parse_args()
    print(vars(args))

    return args


if __name__ == "__main__":
    main_params = get_parser()
    main(main_params)
