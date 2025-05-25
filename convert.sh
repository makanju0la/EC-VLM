#Convert a Vision Language data from NL captions to EC tokens. 
    #Train an EC speaker agent with sequence length 15 and vocab size of 4035
python convert_nl2ec.py --vocab_size 4035 \
--seq_len 15 --load /path/to/speaker/model \
-- in_split /path/to/nl/dataset
-- outfile /path/to/save/converted/ec/data
