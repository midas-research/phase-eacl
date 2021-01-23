# PHASE: Learning Emotional Phase-aware Representations for Suicide Ideation Detection on Social Media

This codebase contains the python scripts for PHASE: Learning Emotional Phase-aware Representations for Suicide Ideation Detection on Social Media.

EACL '21 paper [coming soon](#)

Baselines adapted from [STATENet](https://github.com/midas-research/STATENet_Time_Aware_Suicide_Assessment) | [paper](https://www.aclweb.org/anthology/2020.emnlp-main.619/)

## Environment & Installation Steps

Python 3.6 & Pytorch 1.5

```bash
pip install -r requirements.txt
```

## Run

Execute the following steps in the same environment:

```bash
cd phase-eacl & python train.py --test
```

## Command Line Arguments

To run different variants of PHASE, perform ablation or tune hyperparameters, the following command-line arguments may be used:

```
usage: train.py [-h] [-lr LEARNING_RATE] [-bs BATCH_SIZE] [-e EPOCHS] [-hd HIDDEN_DIM] [-ed EMBEDDING_DIM] [-n NUM_LAYER] [-d DROPOUT] [--current] [--base-model {historic-current,historic,current}] [--model {phase,bilstm,bilstm-attention}] [-t] [--data-dir DATA_DIR] [--random]

optional arguments:
  -h, --help            show this help message and exit
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning Rate for the model (default: 0.0001)
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        batch size (default: 64)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to run (default: 50)
  -hd HIDDEN_DIM, --hidden-dim HIDDEN_DIM
                        Dimension of the hidden state (default: 512)
  -ed EMBEDDING_DIM, --embedding-dim EMBEDDING_DIM
                        Dimension of the encoding (default: 768)
  -n NUM_LAYER, --num-layer NUM_LAYER
                        Number of LSTM layers (default: 1)
  -d DROPOUT, --dropout DROPOUT
                        Dropout probability (default: 0.5)
  --current             Do not concatenate tweet to be assessed with the historical tweets (default: True)
  --base-model {historic-current,historic,current}
                        base model type (default: historic-current)
  --model {phase,bilstm,bilstm-attention}
                        model type (default: phase)
  -t, --test            run on test set (default: False)
  --data-dir DATA_DIR   data directory (default: data)
  --random              random order of historical tweets (default: False)

```

## Dataset Format

Processed dataset format should be a DataFrame as a .pkl file having the following columns:

1. label : 0 or 1 for denoting a tweet as non-suicidal or suicidal respectively.
2. curr_enc : 768-dimensional encoding of the current tweet as a list. (PHASE uses BERT embeddings fine-tuned on EmoNet[1] for the current tweet)
3. enc : list of lists consisting of 768-dimensional encoding for each historical tweet. (PHASE uses BERT embeddings fine-tuned on EmoNet[1])
4. hist_dates : list containing the datetime objects corresponding to each historical tweet.

## Cite

If our work was helpful in your research, please kindly cite this work:

```
@inproceedings{sawhney2021phase,
  title={PHASE: Learning Emotional Phase-aware Representations for Suicide Ideation Detection on Social Media},
  author={
    Sawhney, Ramit and
    Joshi, Harshit and
    Flek, Lucie and
    Shah, Rajiv Ratn
  },
  booktitle={Proceedings of the 16th Conference of the {E}uropean Chapter of the Association for Computational Linguistics},
  year={2021}
}
```

## Ethical Considerations

The preponderance of the work presented in our discussion presents heightened ethical challenges.
As explored in [5], we address the trade-off between privacy and effectiveness.
While data is essential in making models like PHASE effective, we must work within the purview of acceptable privacy practices to avoid coercion and intrusive treatment.
We believe that intervention is a critical step, and PHASE should be used in conjunction with clinical professionals.
To that end, we utilize publicly available Twitter data in a purely observational, and non-intrusive manner.
All tweets shown as examples in our paper and example data have been paraphrased as per the moderate disguise scheme suggested in [4] to protect the privacy of individuals, and attempts should not be made to reverse identify individuals.
Assessments made by PHASE are sensitive and should be shared selectively to avoid misuse, such as Samaritan's Radar.
Our work does not make any diagnostic claims related to suicide.
We study the social media posts in a purely observational capacity and do not intervene with the user experience in any way.

### Note on data

In this work we utilize data from prior work [1, 2].
In compliance with Twitter's privacy guidelines, and the ethical considerations discussed in prior work [2] on suicide ideation detection on social media data, we redirect researchers to theprior work that introduced Emonet [1] and the suicide ideation Twitter dataset [2] to request access to the data.

Please follow the below steps to preprocess the data before feeding it to PHASE:

1. Obtain tweets from Emonet [1], or any other (emotion-based) dataset, to fine-tune a pretrained transformer model (we used BERT-base-cased; English). For Emonet, the authors share the tweet IDs in their dataset (complying to Twitter's privacy guidelines). These tweets then have to be hydrated for further processing.

2. Alternatively, any existing transformer can be used.

3. Using this pretrained transformer, encode all *historical* tweets to obtain a embeddings per historical tweet.

4. For the tweets to be assessed (for which we want to assess suicidal risk), encode the tweets using pretrained encoder (We use fine-tuned BERT on Emonet) to obtain an embedding per tweet to be assessed.
the data provided is a small sample of the original dataset and hence the results obtained on this sample are not fully representative of the results that are obtained on the full dataset.

5. Using these embeddings, create a dataset file in the format explained above under the data directory.

6. We provide the sample format in data/samp_data.pkl

### References

[1] Abdul-Mageed, Muhammad, and Lyle Ungar. "Emonet: Fine-grained emotion detection with gated recurrent neural networks." Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1: Long papers). 2017.

[2] Sawhney, Ramit, Prachi Manchanda, Raj Singh, and Swati Aggarwal. "A computational approach to feature extraction for identification of suicidal ideation in tweets." In Proceedings of ACL 2018, Student Research Workshop, pp. 91-98. 2018.

[3] Reimers, Nils, and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3973-3983. 2019.

[4] Bruckman, A., 2002. Studying the amateur artist: A perspective on disguising data collected in human subjects research on the Internet. Ethics and Information Technology, 4(3), pp.217-231

[5] Glen Coppersmith, Ryan Leary, Patrick Crutchley, andAlex Fine. 2018. Natural language processing of so-cial media as screening for suicide risk.BiomedicalInformatics Insights, 10:117822261879286