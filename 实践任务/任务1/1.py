import os
import sys
import getopt
import uuid
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

#
# Batch Embedding
#
class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_size, embedding, ignore_index=- 100):
        super(EmbeddingLayer, self).__init__()
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.ignore_index = ignore_index

    def forward(self, input):
        '''

        :param input: shape [batch_size, sequence size]
        :return:  shape [batch_size, sequence size, embedding size]
        '''
        batch_size = input.shape[0]
        seq_size = input.shape[1]
        out = torch.zeros(batch_size, seq_size, self.embedding_size)
        for i in range(batch_size):
            for j in range(seq_size):
                if self.ignore_index == input[i][j]:
                    out[i][j] = self.embedding(torch.LongTensor([1]))
                else:
                    out[i][j] = self.embedding(input[i][j])
        return out


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=1, num_layer=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layer = num_layer

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def load(self, path):
        ckp = torch.load(path)
        self.load_state_dict(ckp['encoder_state_dict'])

    def forward(self, input, hidden):
        '''

        :param input:  shape [Batch size, Sequence size, embedding size]
        :param hidden: shape [Batch size, Layer size, hidden size]
        :return:
        '''
        if len(input.shape) != 3:
            raise ValueError(f'input shape {input.shape} is not in the format [batch, seq, embedding]')

        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layer, self.batch_size, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size=1, num_layer=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layer = num_layer

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def load(self, path):
        ckp = torch.load(path)
        self.load_state_dict(ckp['decoder_state_dict'])

    def forward(self, input, hidden):
        '''

        :param input: shape [batch size, sequence size, input size]
        :param hidden:  shape [batch size, num layer, hidden size]
        :return:shape [batch size, sequence size, output size]
        '''
        output = F.relu(input)
        # output shape: [batch size, sequence size, hidden size]
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layer, self.batch_size, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, max_length, batch_size=1, num_layer=1):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.max_length = max_length

        self.attn = nn.Linear(input_size + hidden_size, max_length)
        self.attn_combine = nn.Linear(input_size + hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def load(self, path):
        ckp = torch.load(path)
        self.load_state_dict(ckp['decoder_state_dict'])

    def forward(self, input, hidden, encoder_output):
        '''

        :param input: [batch size, sequence length, input size]
        :param hidden:  [num layer, batch size, hidden size]
        :param encoder_output: [batch size, sequence length, embedding size]
        :return: []
        '''
        attn_tensor = self.attn(torch.cat((input, hidden.permute(1,0,2)), dim=2))

        # [batch size, sequence length, weight size]
        attn_weights = F.softmax(attn_tensor, dim=2)

        # [batch size, sequence length, weight size] * [batch size, embedding size, sequence length] = [batch size, sequence length, attn size]
        attn_applied = torch.bmm(attn_weights, encoder_output)

        # [batch size, sequence length, input size+attn size]  -> [batch size, sequence length, hidden_size]
        combined = self.attn_combine(torch.cat((input, attn_applied), dim=2))

        output = F.relu(combined)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        #output = F.log_softmax(output, dim=2)
        return output, hidden


def pad_collate_fn(batch):
    '''

    :param batch: batch (List[e, l]):
    :return:
    '''
    if len(batch) != batch_size:
        n = batch_size - len(batch)
        last_one = batch[-1]
        for i in range(n):
            batch.append(last_one)

    seq_index_vectors, label_vectors = zip(
        *batch
        # *[(t[0], t[1]) for t in batch]
    )
    return torch.stack(seq_index_vectors), torch.stack(label_vectors)


class Trainer:
    def __init__(self, ds, embedding_layer, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
        self.ds = ds
        self.dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion
        self.latest_acc = 0.

    def train_epoch(self):
        self.encoder.train()
        self.decoder.train()
        for input, target in self.dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss = 0

            # convert np array into torch Tensor
            input_tensor = torch.LongTensor(input)
            embedded_input = self.embedding_layer(input_tensor)
            target_tensor = torch.LongTensor(target)

            encoder_hidden = encoder.init_hidden()
            encoder_output, encoder_hidden = encoder(embedded_input, encoder_hidden)

            decoder_hidden = encoder_hidden
            # shape [batch_size, embedding_size]
            decoder_input = self.embedding_layer(torch.zeros(batch_size, 1, dtype=torch.long))

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                for di in range(ds.label_list_maxlen):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                    loss += self.criterion(decoder_output.squeeze(1), target_tensor[:, di])
                    decoder_input = self.embedding_layer(target_tensor[:, di].unsqueeze(1))
            else:
                for di in range(ds.label_list_maxlen):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = self.embedding_layer(
                        topi.squeeze(1).long().detach())  # detach from history as input
                    # squeeze the seq dimension
                    loss += self.criterion(decoder_output.squeeze(1), target_tensor[:, di])

            print('loss: ', loss.item())
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def evaluate(self, epoch, print_log=False):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            correct = 0
            total = len(ds.expression_list)
            for e, l in zip(ds.expression_list, ds.label_list):
                input = ['' if i == 0 or i == 1 or i == -100 else ds.int2char[i] for i in e]
                input = ''.join(input)
                label = ['' if i == 0 or i == 1 or i == -100 else ds.int2char[i] for i in l]
                label = ''.join(label)

                output = self.predict(input)
                output = ''.join(output)

                if print_log is True:
                    print(f'{input}={output} \t\t\t\t', '√' if output == label else 'x')

                if output == label:
                    correct += 1

            acc = correct / total
            self.latest_acc = acc
            print('[epoch %d ]total accuracy %.4f, encoder optim lr:%.4f, decoder optim lr:%.4f' % (
                epoch, acc, self.encoder_optimizer.param_groups[0]["lr"], self.decoder_optimizer.param_groups[0]["lr"]))
            return acc

    def test(self, print_log=True):
        self.encoder.eval()
        self.decoder.eval()

        formula_table = []
        expression_list = []
        label_list = []

        for a in range(50):
            for b in range(50):
                formula_table.append(f'{a}+{b}={a + b}')

        for line in formula_table:
            pieces = line.split('=')
            expression_list.append(pieces[0])
            label_list.append(pieces[1])

        expression_list_maxlen = len(max(expression_list, key=len))
        label_list_maxlen = len(max(label_list, key=len))

        # padding with space
        for i in range(len(expression_list)):
            while len(expression_list[i]) < expression_list_maxlen:
                expression_list[i] += ' '

        for i in range(len(label_list)):
            while len(label_list[i]) < label_list_maxlen:
                label_list[i] += ' '

        # encoding character into index array
        for i in range(len(expression_list)):
            expression_list[i] = [0, *[1 if c == ' ' else ds.char2int[c] for c in expression_list[i]], 1]
        expression_list_maxlen += 2

        for i in range(len(label_list)):
            # -100 will be ignored by CrossEntropyLoss
            label_list[i] = [0, *[-100 if c == ' ' else ds.char2int[c] for c in label_list[i]], -100]
        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                if label_list[i][j] == -100:
                    # set the first -100 as EOS(1) and break
                    label_list[i][j] = 1
                    break
        label_list_maxlen += 2

        with torch.no_grad():
            correct = 0
            total = len(expression_list)
            for e, l in zip(expression_list, label_list):
                input = ['' if i == 0 or i == 1 or i == -100 else ds.int2char[i] for i in e]
                input = ''.join(input)
                label = ['' if i == 0 or i == 1 or i == -100 else ds.int2char[i] for i in l]
                label = ''.join(label)

                output = self.predict(input)
                output = ''.join(output)

                if print_log is True:
                    print(f'{input}={output} \t\t\t\t', '√' if output == label else 'x')

                if output == label:
                    correct += 1

            acc = correct / total
            self.latest_acc = acc
            print('total accuracy %.4f' % (acc))
            return acc

    def predict(self, text):
        with torch.no_grad():
            # encoding character into index array
            text_index = [0, *[1 if c == ' ' else self.ds.char2int[c] for c in text], 1]
            text_index_len = len(text_index)
            if len(text_index) < ds.expression_list_maxlen:
                for i in range(ds.expression_list_maxlen-len(text_index)):
                    text_index.append(1)

            input_tensor = torch.LongTensor(text_index).view(1, -1).repeat(batch_size, 1)
            embedded_input = self.embedding_layer(input_tensor)

            encoder_hidden = self.encoder.init_hidden()
            encoder_output, encoder_hidden = self.encoder(embedded_input, encoder_hidden)

            decoder_hidden = encoder_hidden
            decoder_input = self.embedding_layer(torch.zeros(batch_size, 1, dtype=torch.long))

            output_index_list = []
            for di in range(20):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                topv, topi = decoder_output.topk(1)

                output_index = topi.squeeze(1).long().detach()
                output_index_list.append(output_index[0].squeeze().item())
                decoder_input = self.embedding_layer(output_index)  # detach from history as input

                if output_index[0].squeeze().item() == 1:
                    break

            return ['' if i == 0 or i == 1 else ds.int2char[i] for i in output_index_list]

    def save_checkpoint(self, epoch, path='.'):
        path = os.path.join(path, 'checkpoint')
        os.makedirs(path, exist_ok=True)

        filename = uuid.uuid4().hex[:10]
        filepath = os.path.join(path, '{0}_epoch{1}_acc{2}.pt'.format(filename, epoch, int(self.latest_acc * 100)))

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_optim_state_dict': self.encoder_optimizer.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'decoder_optim_state_dict': self.decoder_optimizer.state_dict()
        }, filepath)

        pfilepath = os.path.join(path,
                                 '{0}_epoch{1}_acc{2}_params.txt'.format(filename, epoch, int(self.latest_acc * 100)))
        with open(pfilepath, 'w') as f:
            f.write(f'embedding_size={embedding_size}\n')
            f.write(f'hidden_size={hidden_size}\n')
            f.write(f'batch_size={batch_size}\n')
            f.write(f'max_length={self.ds.expression_list_maxlen}\n')
            f.write(f'output_size={len(self.ds.char2int)}\n')
            f.write(f'teacher_forcing_ratio={teacher_forcing_ratio}\n')
            f.write(f'init lr={lr}\n')
            f.write(f'encoder_optimizer lr={self.encoder_optimizer.param_groups[0]["lr"]}\n')
            f.write(f'decoder_optimizer lr={self.decoder_optimizer.param_groups[0]["lr"]}\n')
            f.write(f'accuracy={self.latest_acc}\n')


class ExpressionDataset(Dataset):
    def __init__(self):
        super(ExpressionDataset, self).__init__()
        formula_table = []
        self.expression_list = []
        self.label_list = []

        for a in range(20):
            for b in range(20):
                formula_table.append(f'{a}+{b}={a + b}')
        self.len = len(formula_table)

        # for a in range(10):
        #     for b in range(10):
        #         formula_table.append(f'{a}+{b}={a + b}')

        chars = set(''.join(formula_table))
        self.int2char = dict(enumerate(chars, start=2))
        self.int2char[0] = 'SOS'  # START OF STRING
        self.int2char[1] = 'EOS'  # END OF STRING
        self.char2int = {char: ind for ind, char in self.int2char.items()}

        for line in formula_table:
            pieces = line.split('=')
            self.expression_list.append(pieces[0])
            self.label_list.append(pieces[1])

        self.expression_list_maxlen = len(max(self.expression_list, key=len))
        self.label_list_maxlen = len(max(self.label_list, key=len))
        print("The longest expression has {} characters; The longest label has {} characters".format(
            self.expression_list_maxlen,
            self.label_list_maxlen))
        # padding with space
        for i in range(len(self.expression_list)):
            while len(self.expression_list[i]) < self.expression_list_maxlen:
                self.expression_list[i] += ' '

        for i in range(len(self.label_list)):
            while len(self.label_list[i]) < self.label_list_maxlen:
                self.label_list[i] += ' '

        # print(self.expression_list)
        # print(self.label_list)

        # encoding character into index array
        for i in range(len(self.expression_list)):
            self.expression_list[i] = [0, *[1 if c == ' ' else self.char2int[c] for c in self.expression_list[i]], 1]
        self.expression_list_maxlen += 2

        for i in range(len(self.label_list)):
            # -100 will be ignored by CrossEntropyLoss
            self.label_list[i] = [0, *[-100 if c == ' ' else self.char2int[c] for c in self.label_list[i]], -100]
        for i in range(len(self.label_list)):
            for j in range(len(self.label_list[i])):
                if self.label_list[i][j] == -100:
                    # set the first -100 as EOS(1) and break
                    self.label_list[i][j] = 1
                    break
        self.label_list_maxlen += 2

        print('encoding character into index array!')
        print(self.expression_list)
        print(self.label_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.LongTensor(self.expression_list[index]), torch.LongTensor(self.label_list[index])

    def get_char_count(self):
        return len(self.char2int)


# Define hyperparameters
embedding_size = 16
hidden_size = 60
lr = 0.03
teacher_forcing_ratio = 0.75
batch_size = 10
ds = ExpressionDataset()

def usage():
    print(
        'train_calculator.py -e <embedding_size> -h <hidden_size> -l <learning_rate> -t <teacher_forcing_ratio> -b <batch_size>')


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "e:h:l:t:b:",
                                   ["embedding_size=", "hidden_size=", "learning_rate=", "teacher_forcing_ratio=",
                                    "batch_size="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-e", "--embedding_size"):
            embedding_size = arg
        elif opt in ("-h", "--hidden_size"):
            hidden_size = arg
        elif opt in ("-b", "--batch_size"):
            batch_size = arg
        elif opt in ("-l", "--learning_rate"):
            lr = arg
        elif opt in ("-t", "--teacher_forcing_ratio"):
            teacher_forcing_ratio = arg
    print('---------------------')
    print(f'embedding_size={embedding_size}')
    print(f'hidden_size=   {hidden_size}')
    print(f'init lr=       {lr}')
    print(f'teacher_forcing_ratio={teacher_forcing_ratio}')
    print(f'batch_size=    {batch_size}')
    print('---------------------')

    dict_size = ds.get_char_count()

    embedding = nn.Embedding(dict_size, embedding_size)
    embedding_layer = EmbeddingLayer(embedding_size=embedding_size, embedding=embedding)
    encoder = EncoderRNN(input_size=embedding_size, hidden_size=hidden_size, batch_size=batch_size)
    decoder = AttnDecoderRNN(input_size=embedding_size, output_size=dict_size, hidden_size=hidden_size,
                             max_length=ds.expression_list_maxlen,
                             batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    encoder_scheduler = StepLR(encoder_optimizer, step_size=2, gamma=0.95)
    decoder_scheduler = StepLR(decoder_optimizer, step_size=2, gamma=0.95)

    trainer = Trainer(ds, embedding_layer, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    n_epochs = 200
    no_improvement = 0
    early_stop = 12
    best_acc = 0.
    epoch = 0

    for epoch in range(n_epochs):
        print('======epoch ', epoch)
        if best_acc > 0.8 and teacher_forcing_ratio > 0.05:
            teacher_forcing_ratio -= 0.01
            print(f'teacher_forcing_ratio={teacher_forcing_ratio}')

        trainer.train_epoch()
        acc = trainer.evaluate(epoch)

        if acc > best_acc:
            best_acc = acc
            no_improvement = 0

            if epoch > 5:
                encoder_scheduler.step()
                decoder_scheduler.step()
        else:
            no_improvement += 1
            print('no improvement: ', no_improvement)

            if acc > 0.9:
                encoder_scheduler.step()
                decoder_scheduler.step()

        if no_improvement == early_stop:
            print('early stop due to no improvement!')
            break

    trainer.save_checkpoint(epoch)

    print('end training!')
    trainer.evaluate(0, print_log=True)