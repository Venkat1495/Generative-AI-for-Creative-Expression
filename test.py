# # import pandas as pd
# #
# #
# #
# # def get_all_sentences(ds):
# #     for item in ds:
# #         yield item['text']
# #
# #
# #
# # ds_raw = pd.read_csv('data/spotify_millsongdata.csv')
# # print(get_all_sentences(ds_raw))
#
# import numpy as np
#
# data = np.genfromtxt('data/spotify_millsongdata.csv', delimiter=',', skip_header=1)
# for item in data:
#     print(item['text'])
from datasets import load_dataset


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

ds_raw = load_dataset('opus_books', f'en-fr', split='train')
print(type(ds_raw))