from pypinyin import lazy_pinyin
from pypinyin.style._utils import get_finals


def _is_chinese_char(char):
    try:
        cp = ord(char)
    except:
        return False
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def read_rhyme_vocab(file_path='./rhyme_finals.txt'):
    chinese_rhyme = []
    chinese_rhyme.append(['<pad>'])
    chinese_rhyme.append(['<cls>'])
    chinese_rhyme.append(['<sep>'])
    chinese_rhyme.append(['<word_space>'])

    with open(file_path, 'r') as f:
        rhymes = f.readlines()
        f.close()
    chinese_rhyme += [i.strip().split(',') for i in rhymes]
    return chinese_rhyme


def get_rhyme(word):
    rhyme = get_finals(lazy_pinyin(word)[0], strict=False)
    rhymes_chinese = read_rhyme_vocab()
    for idx, rhymes in enumerate(rhymes_chinese):
        if rhyme in rhymes:
            return idx
    return len(rhymes_chinese)


def get_sent_rhyme(sentence):
    rhy_index = []
    for word in sentence:
        rhy_index.append(get_rhyme(word))
    return rhy_index


def get_rhyme_vocab():
    chinese_rhyme = (read_rhyme_vocab())

    rhyme_vocab = {
        i: rhyme for i, rhyme in enumerate(chinese_rhyme)
    }
    return rhyme_vocab

