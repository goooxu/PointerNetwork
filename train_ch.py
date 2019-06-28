import torch
import torch.utils.data
import numpy
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Options:
    
    def __init__(self, 
    training_data_filename, 
    test_data_filename,
    input_size, 
    embedding_size, 
    rnn_hidden_size, 
    attention_hidden_size, 
    amplification, 
    perturbation, 
    training_batch_size, 
    test_batch_size, 
    total_epochs, 
    checkpoint=None):
        self.training_data_filename = training_data_filename
        self.test_data_filename = test_data_filename
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.rnn_hidden_size = rnn_hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.amplification = amplification
        self.perturbation = perturbation
        self.training_batch_size = training_batch_size
        self.test_batch_size = test_batch_size
        self.total_epochs = total_epochs
        self.checkpoint = checkpoint

PAD = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, math.sin(0.0), math.sin(0.0)])

class Dataset(torch.utils.data.Dataset):

    def __init__(self, filename, perturbation=0):
        self.X_seq_list = []
        self.Y_seq_list = []
        self._read_data(filename)
        self._perturb_data(perturbation)

    def _read_data(self, filename):
        f = open(filename, 'r')
        for line in f:
            columns = line.split()
            separator = columns.index('output')
            X_seq = [float(i) for i in columns[:separator]]
            X_seq = [i for i in zip(X_seq[::2], X_seq[1::2])]
            X_seq = [(
                i[0], 
                i[1], 
                i[0] * i[0], 
                i[1] * i[1],
                i[0] * i[1],
                math.sin(i[0] * math.pi / 2),
                math.sin(i[1] * math.pi / 2)) for i in X_seq]
            Y_seq = [int(i) for i in columns[separator + 1:]]

            self.X_seq_list.append(X_seq)
            self.Y_seq_list.append(Y_seq)

    def _perturb_data(self, perturbation):
        X_seq_list = []
        Y_seq_list = []
        
        for X_seq, Y_seq in zip(self.X_seq_list, self.Y_seq_list):
            X_seq_list.append(numpy.array(X_seq))
            Y_seq_list.append(numpy.array(Y_seq))

            for i in range(perturbation):
                perm = numpy.random.permutation(len(X_seq))
                permuted_X_seq = numpy.empty((len(X_seq), 7))
                permuted_Y_seq = numpy.empty(len(Y_seq))

                for i in range(len(X_seq)):
                    permuted_X_seq[perm[i]] = X_seq[i]
                
                for i in range(len(Y_seq)):
                    permuted_Y_seq[i] = perm[Y_seq[i] - 1] + 1

                m = min(permuted_Y_seq)
                permuted_Y_seq = permuted_Y_seq[:-1]
                while permuted_Y_seq[0] != m:
                    permuted_Y_seq = numpy.roll(permuted_Y_seq, 1)
                permuted_Y_seq = numpy.append(permuted_Y_seq, permuted_Y_seq[0])

                X_seq_list.append(permuted_X_seq)
                Y_seq_list.append(permuted_Y_seq)
        
        self.X_seq_list = X_seq_list
        self.Y_seq_list = Y_seq_list

    def __len__(self):
        return len(self.X_seq_list)

    def __getitem__(self, idx):
        X_seq = self.X_seq_list[idx]
        Y_seq = self.Y_seq_list[idx]
        return numpy.concatenate(([PAD], X_seq)), Y_seq

def collate_fn(data):
    X, Y = zip(*data)

    max_length = len(max(X, key=len))
    X = [numpy.concatenate((i, numpy.array([PAD] * (max_length - len(i))).reshape(-1, 7))) for i in X]

    max_length = len(max(Y, key=len))
    Y = [numpy.concatenate((i, numpy.array([0] * (max_length - len(i))))) for i in Y]

    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.long, device=device)
    return X, Y

class PointerNetwork(torch.nn.Module):
    
    def __init__(self, input_size, embedding_size, rnn_hidden_size, attention_hidden_size):
        super(PointerNetwork, self).__init__()
        
        self.fc0 = torch.nn.Linear(input_size, embedding_size, bias=False)
        self.encoder = torch.nn.LSTM(embedding_size, rnn_hidden_size, bias=False, batch_first=True)
        self.decoder = torch.nn.LSTMCell(embedding_size, rnn_hidden_size, bias=False)
        self.fc1 = torch.nn.Linear(rnn_hidden_size, attention_hidden_size, bias=False)
        self.fc2 = torch.nn.Linear(rnn_hidden_size, attention_hidden_size, bias=False)
        self.fc31 = torch.nn.Linear(attention_hidden_size, attention_hidden_size, bias=False)
        self.fc32 = torch.nn.Linear(attention_hidden_size, attention_hidden_size, bias=False)
        self.fc33 = torch.nn.Linear(attention_hidden_size, attention_hidden_size, bias=False)
        self.fc4 = torch.nn.Linear(attention_hidden_size, 1, bias=False)
        
    def forward(self, X, training, Y, amplification=None): # X:(batch, X_seq_length, input_size), Y:(batch, Y_seq_length)
        batch_size = X.size(0)
        Y_seq_length = Y.size(1)

        X = self.fc0(X) # (batch, X_seq_length, embedding_size)
        
        encoder_outputs, (encoder_hidden_state, encoder_cell_state) = self.encoder(X) # encoder_outputs:(batch, X_seq_length, rnn_hidden_size), encoder_hidden_state:(1, batch, rnn_hidden_size), encoder_cell_state:(1, batch, rnn_hidden_size)
        encoder_outputs = encoder_outputs.transpose(1, 0) # (X_seq_length, batch, rnn_hidden_size)
        encoder_outputs = self.fc1(encoder_outputs)  # (X_seq_length, batch, attention_hidden_size)
        
        hidden_state = encoder_hidden_state[0] # (batch, rnn_hidden_size)
        cell_state = encoder_cell_state[0] # (batch, rnn_hidden_size)
        decoder_input = X[:, 0] # (batch, embedding_size)

        if training:

            probabilities = []

            for i in range(Y_seq_length):
                hidden_state, cell_state = self.decoder(decoder_input, (hidden_state, cell_state)) # hidden_state:(batch, rnn_hidden_size), cell_state:(batch, rnn_hidden_size)
                decoder_output = self.fc2(hidden_state) # (batch, attention_hidden_size)
                        
                blended_output = torch.tanh(encoder_outputs + decoder_output) # (X_seq_length, batch, attention_hidden_size)
                blended_output = torch.tanh(self.fc31(blended_output)) # (X_seq_length, batch, attention_hidden_size)
                blended_output = torch.tanh(self.fc32(blended_output)) # (X_seq_length, batch, attention_hidden_size)
                blended_output = torch.tanh(self.fc33(blended_output)) # (X_seq_length, batch, attention_hidden_size)
                
                probability = self.fc4(blended_output).squeeze(2) # (X_seq_length, batch)
                probability = probability.transpose(0, 1) # (batch, X_seq_length)
                probability = torch.nn.functional.log_softmax(probability, dim=1) # (batch, X_seq_length)
                probabilities.append(probability)

                indices = Y[..., i] # (batch)
                decoder_input = X[range(batch_size), indices] # (batch, embedding_size)

            probabilities = torch.stack(probabilities, dim=1) # (batch, Y_seq_length, X_seq_length)
            indices = probabilities.argmax(dim=2) # (batch, Y_seq_length)
            return probabilities, indices

        else:

            probability_groups = [] # list(batch * amplification, X_seq_length)
            row_indices = [] # list(batch, amplification)
            column_indices = [] #list(batch, amplification)

            for i in range(Y_seq_length):

                hidden_state, cell_state = self.decoder(decoder_input, (hidden_state, cell_state)) # hidden_state:(batch * amplification, rnn_hidden_size), cell_state:(batch * amplification, rnn_hidden_size)
                decoder_output = self.fc2(hidden_state) # (batch * amplification, attention_hidden_size)

                amplifying_encoder_outputs = encoder_outputs[:, torch.arange(batch_size).repeat_interleave(1 if i == 0 else amplification)] # (X_seq_length, batch * amplification, attention_hidden_size)

                blended_output = torch.tanh(amplifying_encoder_outputs + decoder_output) # (X_seq_length, batch * amplification, attention_hidden_size)
                blended_output = torch.tanh(self.fc31(blended_output)) # (X_seq_length, batch * amplification, attention_hidden_size)
                blended_output = torch.tanh(self.fc32(blended_output)) # (X_seq_length, batch * amplification, attention_hidden_size)
                blended_output = torch.tanh(self.fc33(blended_output)) # (X_seq_length, batch * amplification, attention_hidden_size)
                
                probability = self.fc4(blended_output).squeeze(2) # (X_seq_length, batch * amplification)
                probability = probability.transpose(0, 1) # (batch * amplification, X_seq_length)
                probability = torch.nn.functional.log_softmax(probability, dim=1) # (batch * amplification, X_seq_length)
                total_probability = probability.view(batch_size, 1 if i == 0 else amplification, -1) # (batch, amplification, X_seq_length)
                
                if i != 0:
                    total_probability = total_probability + accumulation_probability.unsqueeze(2) # (batch, amplification, X_seq_length)

                v1, i1 = total_probability.topk(amplification, dim=2) # (batch, amplification, amplification)

                v1 = v1.view(batch_size, -1) # (batch, amplification * amplification)
                i1 = i1.view(batch_size, -1) # (batch, amplification * amplification)
                
                accumulation_probability, i2 = v1.topk(amplification, dim=1) # accumulation_scores:(batch, amplification), i2:(batch, amplification)

                irow = i2 // amplification # (batch, amplification)
                icolumn = i1[torch.arange(batch_size).unsqueeze(1), i2] # (batch, amplification)

                row_indices.append(irow)
                column_indices.append(icolumn)

                irow_flat = irow + torch.arange(0, batch_size * amplification, amplification, device=device).unsqueeze(1) # (batch * amplification)
                irow_flat = irow_flat.view(-1) // (amplification if i == 0 else 1) # (batch * amplification)

                hidden_state = hidden_state[irow_flat] # (batch * amplification, rnn_hidden_size)
                cell_state = cell_state[irow_flat] # (batch * amplification, rnn_hidden_size)
                probability_group = probability[irow_flat].view(batch_size, amplification, -1) # (batch, amplification, X_seq_length)
                probability_groups.append(probability_group) 

                decoder_input = X[torch.arange(batch_size).unsqueeze(1), icolumn] # (batch, amplification, embedding_size)
                decoder_input = decoder_input.view(batch_size * amplification, -1) # (batch * amplification, embedding_size)

            row_indices = torch.stack(row_indices) # (Y_seq_length, batch, amplification)
            probability_groups = torch.stack(probability_groups) # (Y_seq_length, batch, amplification, X_seq_length)
            column_indices = torch.stack(column_indices) # (Y_seq_length, batch, amplification)

            probabilities = []
            indices = []

            j = torch.zeros(batch_size, dtype=torch.long) # (batch)
            for i in reversed(range(Y_seq_length)):
                probabilities.append(probability_groups[i][range(batch_size), j])
                indices.append(column_indices[i][range(batch_size), j])
                j = row_indices[i][range(batch_size), j] # (batch)

            probabilities = torch.stack(probabilities[::-1], dim=1) # (batch, Y_seq_length, X_seq_length)
            indices = torch.stack(indices[::-1], dim=1) # (batch, Y_seq_length)
            return probabilities, indices

def trainOneEpoch(dataLoader, model, training, optimizer=None, amplification=None):
    average_loss = torch.tensor(0.0)
    batch_count = 0
    correct_samples = 0
    total_samples = 0

    for samples in dataLoader:
    
        X_samples = samples[0] # (batch, X_seq_length, input_size)
        Y_samples = samples[1] # (batch, Y_seq_length)
        X_seq_length = X_samples.size(1)

        probabilities, indices = model(X_samples, training, Y_samples, amplification) # probabilities:(batch, Y_seq_length, X_seq_length), indices:(batch, Y_seq_length)
        correct_samples += sum([1 if torch.equal(x.data, y.data) else 0 for x, y in zip(indices, Y_samples)])
        total_samples += Y_samples.size(0)

        P_samples = probabilities.view(-1, X_seq_length) # (batch * Y_seq_length, X_seq_length)
        Y_samples = Y_samples.view(-1) # (batch * Y_seq_length)
        loss = torch.nn.functional.nll_loss(P_samples, Y_samples) # (1)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss += loss.detach()
        batch_count += 1

    average_loss /= batch_count
    
    return average_loss, correct_samples / total_samples

def train(options):
    print(options.__dict__)

    trainingDataset = Dataset(options.training_data_filename, options.perturbation)
    testDataset = Dataset(options.test_data_filename)
    trainingDataLoader = torch.utils.data.DataLoader(trainingDataset, shuffle=True, collate_fn=collate_fn, batch_size=options.training_batch_size)
    testDataLoader = torch.utils.data.DataLoader(testDataset, shuffle=True, collate_fn=collate_fn, batch_size=options.test_batch_size)

    model = PointerNetwork(
        options.input_size, 
        options.embedding_size, 
        options.rnn_hidden_size, 
        options.attention_hidden_size).to(device)
    print(model.parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3**(1/options.total_epochs))

    last_epoch = 0
    if options.checkpoint != None:
        checkpoint = torch.load(options.checkpoint)
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Loaded checkpoint, last_epoch=', last_epoch)

    for epoch in range(last_epoch + 1, options.total_epochs + 1):

        model.train()
        training_loss, training_accuracy = trainOneEpoch(trainingDataLoader, model, True, optimizer=optimizer)

        model.eval()
        with torch.no_grad():
            test_loss_beam_search, test_accuracy_beam_search = trainOneEpoch(testDataLoader, model, False, amplification=options.amplification)
            test_loss, test_accuracy = trainOneEpoch(testDataLoader, model, False, amplification=1)

        lr = optimizer.param_groups[0]['lr']

        scheduler.step()

        print('Epoch: {:3}, LR: {:.3e}, Training: ({:.3f}, {:.1f}%), Test: ({:.3f}, {:.1f}%) ({:.3f}, {:.1f}%)'.format(
            epoch, 
            lr,
            training_loss.item(),
            training_accuracy * 100, 
            test_loss_beam_search.item(),
            test_accuracy_beam_search * 100,
            test_loss.item(),
            test_accuracy * 100))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, 'models/ch_epoch_{}'.format(epoch))

if __name__ == '__main__':
    train(Options(
        training_data_filename='data/ch_all_training.txt',
        test_data_filename='data/ch_all_test.txt',
        input_size=7,
        embedding_size=128, 
        rnn_hidden_size=128,
        attention_hidden_size=128, 
        amplification=3,
        perturbation=4,
        training_batch_size=500, 
        test_batch_size=50,
        total_epochs=100))