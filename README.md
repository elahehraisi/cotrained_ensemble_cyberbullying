# requirements

- python 3
- sklearn, numpy, scipy, pandas, json
- gensim (to train doc2vec and word2vec)
- pytorch
- CUDA

# Description
This repository has the scripts for weakly supervised machine learning for cyberbullying detection in social media. Two methodology are introduced: 1)Participant-Vocabulary Consistency (PVC); 2) Co-trained Ensembles of Embedding Models


# Participant-Vocabulary Consistency (PVC)

- script ‘main_pvc.py’ is created to train pvc model.

- To run the script ‘main_pvc.py’, three files are required in ‘/data/data_set/files/’ directory:

	1) The data file containing interactions. In this example, we used synthetic dataset, but you could used any dataset (Twitter/Instagram/etc.). The data file is in json format. For each interaction, three information is required: user_name, target_name, and text like below:
        ```
        {"user_name": "1", "target_name": "2", "text": "ugly friend word1 word11 word111"}
        ```

        There is sample synthetic dataset in ‘/data/data_set/files/’ directory. 

	2) All the bigrams under the name ’bigrams.txt’ 

	3) A file ‘badwords.txt’ containing the seed words

- the output of ‘main_pvc.py’ will be two files which will be saved in ‘/data/data_set/saved_data’
	1) results.txt: sorted words from high to low based on their given vocabulary score 
	2) PVC_bullying_interaction.txt: user pairs and their interaction score based on pvc. User pairs are sorted from high to low.

for more information please see the paper ["Cyberbullying Detection with Weakly Supervised Machine Learning"](http://people.cs.vt.edu/~bhuang/papers/raisi-asonam17.pdf)

# Co-trained Ensembles of Embedding Models

- the main script is ‘main_cotrained_ensemble.py’, but one needs to run ‘preprocess.py’ script to generate the required files.

- we introduced 12 separate models which are the combination of four message learners and three user learners. Message learners are: doc2vec, BoW, embedding, and rnn. User learners are: node2vec, embedding, and none (none means not using user learner)

- In the 'preprocess.py' script, there are six sections. We recommend running each section in isolation by commenting the other sections.
	*  At the end of running sections 1 and 2, you have created separate files for training and validation. They are saved in ‘/data/data_set/files/’ directory (all messages will be shuffled and 80% will be for training, and 20% will be for validation). 

	* in section 3, lower and upper bound for each message will be computed based negative words and positive words. List of negative and positive words are in ‘/data/data_set/files/’ as bad words.txt and positive-words.txt. The resulting upper and lower bounds will be saved in ‘./saved_models/data_set/’

	* in section 4, 5, and 6 word2vec, doc2vec, and node2vec will be trained respectively. We used gensim to train word2vec and node2vec. To train node2vec, we download the files from SNAP webpage (https://snap.stanford.edu/node2vec/). The resulting files will be saved in ‘./saved_models/data_set/’.

- The ‘main_cotrained_ensemble.py’ script accepts the following arguments:
	 ```
    --message_classifier: which message classifier you would like to use (doc2vec/rnn/emb/bow)
	-- user_classifier: which user classifier you would like to use (node2vec/emb/none)
	-- batch_normalization: bn/no-bn  if bn: batch normalization layer is used. if no-bn: no batch normalization is used
	-- data_set: name of dataset (e.g. Twitter/Instagram/etc.) we used synthetic dataset in this example.
	-- data:location of the data corpus
	-- rnn_model: type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU) we only used LSTM
	-- emsize: size of word embeddings
	-- useremsize: size of user embeddings
	-- checkpoint: define the point we check the validation set to save the model
	-- patient: the number of times we will pursue training even if validation error is increasing
	-- nhid: number of hidden units per layer
	-- nlayers: number of layers
	-- lr: initial learning rate
	-- epochs: upper epoch limit
	-- batch_size: batch size
	-- dropout: dropout applied to layers
	-- tied: tie the word embedding and softmax weights
	-- cuda: use CUDA
	-- d_in_user: input dimension for users; it is created based on node2vec
	-- d_out_user: output dimension for users; it is one since we have binary classifier
	-- optimizer: optimizer type
	-- d_in_msg: input dimension for message
	-- d_out_msg: output dimension for message
    ```
    With these arguments, a variety of models can be run. As an example you could train co-trained ensemble of doc2vec message learner and node2vec user learner using batch normalization on synthetic dataset:
    ```
    python main_cotrained_ensemble.py --cuda --message_classifier ‘doc2vec’ --user_classifier ‘node2vec’ --batch_normalization ‘bn’ --data_set ‘synthetic’
    ```
- After running ‘main_cotrained_ensemble.py’, the final message and user learners will be saved in ‘/data/data_set/saved_data’.

- Script ‘load_model_extract_conversation.py’ is to load the last saved message and user learners, then compute the bullying score of user pairs, and finally sort them from high to low based on bullying score, and save them in ‘user_pair_indices.txt’ file.

for more information please see the paper ["Weakly Supervised Cyberbullying Detection usingCo-trained Ensembles of Embedding Models"](http://people.cs.vt.edu/~bhuang/papers/raisi-asonam18.pdf)




