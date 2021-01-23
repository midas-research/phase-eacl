import datetime

import numpy as np
import torch
import torch.nn as nn
import transformers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def fetch_vectors(string_list, batch_size=64, max_len=142):
    DEVICE = torch.device("cuda")
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.to(DEVICE)

    fin_features = []
    for data in tqdm(chunks(string_list, batch_size), total=np.ceil(len(string_list) / batch_size)):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:140])
            #             print(type(x), x.shape, x)
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:max_len])

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    fin_features = np.vstack(fin_features)
    return fin_features


def fetch_sentence_vectors(sentences):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings


def pad_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]

    lens = [len(x) for x in data]

    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    tweet = torch.tensor(tweet)
    lens = torch.tensor(lens)

    return [target, tweet, data, lens]


def pad_ts_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]
    timestamp = [item[3] for item in batch]

    lens = [len(x) for x in data]

    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    timestamp = nn.utils.rnn.pad_sequence(timestamp, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    tweet = torch.tensor(tweet)
    lens = torch.tensor(lens)

    return [target, tweet, data, lens, timestamp]


def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append(datetime.datetime.timestamp(t))

    np.array(timestamp) - timestamp[-1]
    return timestamp
