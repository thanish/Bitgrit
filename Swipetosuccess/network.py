import torch.nn as nn

class RNN_for_Text(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):

        #embedded = [sent len, batch size, emb dim]
        embedding = x.view(1, -1, self.embedding_dim)
        
        output, hidden = self.rnn(embedding)
        
        #output = [sent len, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
        # assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        out = self.fc(hidden)
        return out
    
    
class NN_for_num_Features(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super().__init__()

        self.relu = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(hidden_dim)
        self.hidden_layer = nn.Linear(num_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        
        out = self.hidden_layer(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    
    
class MulticlassClassification(nn.Module):
    def __init__(self, num_features, num_labels):
        super(MulticlassClassification, self).__init__()
        
        self.num_features = num_features
        self.num_labels = num_labels
        
        self.hidden_layer_1 = nn.Linear(self.num_features, 2056)
        self.hidden_layer_2 = nn.Linear(2056, 2056)
        self.output = nn.Linear(2056, self.num_labels)
        
        self.relu = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(2056)
        self.batch_norm_2 = nn.BatchNorm1d(2056)

        
    def forward(self, X):
        out = self.hidden_layer_1(X)
        out = self.batch_norm_1(out)
        out = self.relu(out)

        out = self.hidden_layer_2(out)
        out = self.batch_norm_2(out)
        out = self.relu(out)
                
        out = self.output(out)
        #out = nn.Softmax()(out)
        
        return out

    
