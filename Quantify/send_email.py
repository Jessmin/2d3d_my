import numpy as np

# # define
# dict = {}
# dict['a'] =1
# # save
# np.save('dict.npy', dict)
# # load
dict_load = np.load('2021-03-01.npy', allow_pickle=True)

print("dict =", dict_load.item().keys())
