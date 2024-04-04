# import pandas as pd
# # #
# # #
# # #
# # # def get_all_sentences(ds):
# # #     for item in ds:
# # #         yield item['text']
# # #
# # #
# # #
# # # ds_raw = pd.read_csv('data/spotify_millsongdata.csv')
# # # print(get_all_sentences(ds_raw))
# #
# # import numpy as np
# #
# # data = np.genfromtxt('data/spotify_millsongdata.csv', delimiter=',', skip_header=1)
# # for item in data:
# #     print(item['text'])
# from datasets import load_dataset
# from tokenizers import Tokenizer
# from tokenizers.models import WordLevel
# from tokenizers.trainers import WordLevelTrainer
# from tokenizers.pre_tokenizers import Whitespace
# from pathlib import Path
# from config import get_config
#
# def get_all_sentences(ds):
#     for item in ds:
#         yield item['text']
#
# def get_or_build_tokenizer(config, ds):
#     # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
#     tokenizer_path = Path(config['tokenizer_file'])
#     if not Path.exists(tokenizer_path):
#         tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
#         tokenizer.pre_tokenizer = Whitespace()
#         trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
#         tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
#         tokenizer.save(str(tokenizer_path))
#     else:
#         tokenizer = Tokenizer.from_file(str(tokenizer_path))
#     return tokenizer
#
#
# ds_raw = pd.read_csv('data/spotify_millsongdata.csv')
# config = get_config()
# tokenizer_src = get_or_build_tokenizer(config, ds_raw)
# outs = []
# for item in ds_raw['text']:
#     outs.append(len(tokenizer_src.encode(item).ids))
#
# max_value = max(outs)  # Finds the maximum value in the outs array
# min_value = min(outs)  # Finds the minimum value in the outs array
# print("Maximum:", max_value)
# print("Minimum:", min_value)
#
# # next_max_value = max(value for value in outs if value != max_value)
# #
# # print("Next Maximum:", next_max_value)
#
# # Count elements in the specified ranges
# print(len(outs))
# count_less_than_100 = len([x for x in outs if x < 100])
# count_100_to_200 = len([x for x in outs if 100 <= x < 200])
# count_200_to_300 = len([x for x in outs if 200 <= x < 300])
# count_300_to_600 = len([x for x in outs if 300 <= x < 600])
# count_600_to_900 = len([x for x in outs if 600 <= x < 900])
# count_900_to_1200 = len([x for x in outs if 900 <= x < 1200])
#
# print("Number of elements less than 100:", count_less_than_100)
# print("Number of elements less than 200:", count_100_to_200)
# print("Number of elements less than 300:", count_200_to_300)
# print("Number of elements from 300 to 600:", count_300_to_600)
# print("Number of elements from 600 to 900:", count_600_to_900)
# print("Number of elements from 900 to 1200:", count_900_to_1200)
#
# total = count_less_than_100 + count_100_to_200 + count_200_to_300 + count_300_to_600 + count_600_to_900 + count_900_to_1200
# print(total)


# Let's calculate the division.
result = 1204 // 299
print(result)
