from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
from torch import nn
import math


class BiLSTMAttn(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim // 2, dropout=dropout if num_layers > 1 else 0,
                               num_layers=num_layers, batch_first=True, bidirectional=True)

    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden

    def forward(self, features, lens):
        features = self.dropout(features)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True, enforce_sorted=False)
        outputs, (hn, cn) = self.encoder(packed_embedded)
        outputs, output_len = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        fbout = outputs[:, :, :self.hidden_dim // 2] + outputs[:, :, self.hidden_dim // 2:]
        fbhn = (hn[-2, :, :] + hn[-1, :, :]).unsqueeze(0)
        attn_out = self.attnetwork(fbout, fbhn)

        return attn_out  # batch_size, hidden_dim/2


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, dropout=dropout, num_layers=num_layers, batch_first=True,
                              bidirectional=True)

    def forward(self, features, lens):
        # print(self.hidden.size())
        features = self.dropout(features)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(features, lens, batch_first=True, enforce_sorted=False)
        outputs, hidden_state = self.bilstm(packed_embedded)
        outputs, output_len = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden_state  # outputs: batch, seq, hidden_dim - hidden_state: hn, cn: 2*num_layer, batch_size, hidden_dim/2


class HistoricCurrent(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, model):
        super().__init__()
        self.model = model
        if self.model == "phase":
            self.historic_model = PHASE(embedding_dim, hidden_dim, conv_size=5, output_dim=1, levels=4,
                                           dropconnect=dropout)
        elif self.model == "bilstm":
            self.historic_model = BiLSTM(embedding_dim, hidden_dim, num_layers, dropout)
        elif self.model == "bilstm-attention":
            self.historic_model = BiLSTMAttn(embedding_dim, hidden_dim, num_layers, dropout)

        self.fc_ct = nn.Linear(768, hidden_dim)
        self.fc_ct_attn = nn.Linear(768, hidden_dim // 2)

        self.fc_concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_concat_attn = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(hidden_dim, 2)

    @staticmethod
    def combine_features(tweet_features, historic_features):
        return torch.cat((tweet_features, historic_features), 1)

    def forward(self, tweet_features, historic_features, lens, timestamp):
        if self.model == "phase":
            outputs, _ = self.historic_model(historic_features, timestamp)
            tweet_features = F.relu(self.fc_ct(tweet_features))
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = F.relu(self.fc_concat(combined_features))
        elif self.model == "bilstm":
            outputs, (h_n, c_n) = self.historic_model(historic_features, lens)
            outputs = torch.mean(outputs, 1)
            tweet_features = F.relu(self.fc_ct(tweet_features))
            # tweet_features = self.dropout(tweet_features)
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = F.relu(self.fc_concat(combined_features))
        elif self.model == "bilstm-attention":
            outputs = self.historic_model(historic_features, lens)
            tweet_features = F.relu(self.fc_ct_attn(tweet_features))
            combined_features = self.combine_features(tweet_features, outputs)
            combined_features = self.dropout(combined_features)
            x = F.relu(self.fc_concat_attn(combined_features))

        x = self.dropout(x)

        return self.final(x), _


class Historic(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.historic_model = BiLSTM(embedding_dim, hidden_dim, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.final = nn.Linear(32, 2)

    def forward(self, tweet_features, historic_features, lens, timestamp):
        outputs, (h_n, c_n) = self.historic_model(historic_features, lens)
        # outputs = torch.mean(outputs, 1)
        hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        x = F.relu(self.fc1(hidden))
        # x = F.relu(self.fc1(outputs))
        return self.final(x)


class Current(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(768, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.final = nn.Linear(32, 2)

    def forward(self, tweet_features, historic_features, lens, timestamp):
        x = F.relu(self.fc1(tweet_features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.final(x)


class PHASE(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_size, output_dim, levels, dropconnect=0., dropout=0., dropres=0.3):
        super(PHASE, self).__init__()

        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.dropres = dropres
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        self.output_dim = output_dim
        self.levels = levels
        self.chunk_size = hidden_dim // levels

        self.kernel = nn.Linear(int(input_dim + 1), int(hidden_dim * 4 + levels * 2))
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(int(hidden_dim + 1), int(hidden_dim * 4 + levels * 2))
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)

        self.nn_scale = nn.Linear(int(hidden_dim), int(hidden_dim // 6))
        self.nn_rescale = nn.Linear(int(hidden_dim // 6), int(hidden_dim))
        self.nn_conv = nn.Conv1d(int(hidden_dim), int(self.conv_dim), int(conv_size), 1)
        self.nn_output = nn.Linear(int(self.conv_dim), int(output_dim))

        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)

    def cumax(self, x, mode='l2r'):
        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x

    def step(self, inputs, c_last, h_last, interval):
        x_in = inputs

        # Integrate inter-visit time intervals
        interval = interval.unsqueeze(-1)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1))

        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r')
        f_master_gate = f_master_gate.unsqueeze(2)
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels * 2], 'r2l')
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels * 2:]
        x_out = x_out.reshape(-1, self.levels * 4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels * 2])
        o_gate = torch.sigmoid(x_out[:, self.levels * 2:self.levels * 3])
        c_in = torch.tanh(x_out[:, self.levels * 3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (
                i_master_gate - overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out

    def forward(self, input, time, device="cuda"):
        batch_size, time_step, feature_dim = input.size()
        c_out = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_out = torch.zeros(batch_size, self.hidden_dim).to(device)

        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(self.conv_size,
                                                                                                  batch_size,
                                                                                                  self.hidden_dim).to(
            device)
        tmp_dis = torch.zeros((self.conv_size, batch_size)).to(device)
        h = []
        origin_h = []
        distance = []
        for t in range(time_step):
            out, c_out, h_out = self.step(input[:, t, :], c_out, h_out, time[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim + self.levels], -1)
            cur_distance_in = torch.mean(out[..., self.hidden_dim + self.levels:], -1)
            # origin_h.append(out[..., :self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)
            distance.append(cur_distance)

            # Re-weighted convolution operation
        local_dis = tmp_dis.permute(1, 0)
        local_dis = torch.cumsum(local_dis, dim=1)
        local_dis = torch.softmax(local_dis, dim=1)
        local_h = tmp_h.permute(1, 2, 0)
        local_h = local_h * local_dis.unsqueeze(1)

        # Re-calibrate Progression patterns
        local_theme = torch.mean(local_h, dim=-1)
        local_theme = self.nn_scale(local_theme)
        local_theme = torch.relu(local_theme)
        local_theme = self.nn_rescale(local_theme)
        local_theme = torch.sigmoid(local_theme)

        local_h = self.nn_conv(local_h).squeeze(-1)
        local_h = local_theme * local_h

        return local_h, torch.stack(distance)
