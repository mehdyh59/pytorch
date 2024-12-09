import io
import os
import unicodedata
import string
import glob
import torch
import random


ALL_LETTERS = string.ascii_letters + ".,;"
N_LETTERS = len(ALL_LETTERS)
DATA_DIRECTORY = '/Users/c3666498/Codes/Torch/pytorch/data/names/*.txt'

# turn a unicode string to ascii
def unicode_to_ascii(s):
    return ''.join(
        c for c  in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )

# build the category-line dictionary, a list of names per language 
def load_data():
    category_lines = {}
    all_categroies = []

    def find_files(path):
        return glob.glob(path)

    # read a file and spilit it into lines
    def read_lines (filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files(DATA_DIRECTORY):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categroies.append(category)

        lines = read_lines(filename)
        category_lines[category]=lines
    
    return category_lines, all_categroies


"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.

To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.

That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""

#find letter index for all_letters 
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

#turn a letter to a < 1 * n_letters> tensor 
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# turn a line into a <line_lenght * 1 * n_letters>
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example( category_lines, all_categroies):

    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categroies)
    line = random_choice(category_lines[category])
    category_tensor = line_to_tensor(line)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


if __name__ == '__main__':
    print(ALL_LETTERS)
    print( unicode_to_ascii("Ślusàrski"))

    category_lines , all_categories = load_data()
    print(f'all categories are: {all_categories}')
    print(category_lines['Italian'][:5])

    print( letter_to_tensor('J'))
    print(line_to_tensor('Jones').size())