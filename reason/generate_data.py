import pathlib
import shutil
import subprocess
import argparse
import random, string
import tqdm
import copy
import os
import re
import time


def sample_substring(string, min, max):
    substring_len = random.randint(min, max)
    init = random.randint(0, len(string) - substring_len)
    return string[init:init + substring_len], init, substring_len


class Pattern:
    def __init__(self, pattern_list):
        self.pattern_list = pattern_list

    def replace(self, to_replace, replace):
        indices = []
        for i, char in enumerate(self.pattern_list):
            if char == to_replace:
                indices.append(i)

        new_pattern = []
        new_pattern.extend(self.pattern_list[:indices[0]])
        if not isinstance(replace, list):
            replace = [replace]
        new_pattern.extend(replace)
        for i, j in zip(indices[:-1], indices[1:]):
            new_pattern.extend(self.pattern_list[i+1:j])
            new_pattern.extend(replace)
        new_pattern.extend(self.pattern_list[indices[-1]+1:])
        return new_pattern



class SubstRule:
    '''
    example:
    addcomm = SubstRule(["A", "B"], "A+B=B+A")
    out = addcomm.rewrite({"A":"xwd239+c", "B":"23oijs_c"})
    xwd239+c+23oijs_c=23oijs_c+xwd239+c
    '''
    def __init__(self, upper_case_letters, pattern):
        self.upper_case_letters = upper_case_letters
        self.pattern = Pattern(pattern)
    def subst(self, substitute):
        out = self.pattern
        for char in self.upper_case_letters:
            out = Pattern(out.replace(char, substitute[char]))
        return out.pattern_list


def gen_subst(upper_case_letters, lower_case_letters, math_symbols):
    substitute = {}
    substitute_str = []
    for char in upper_case_letters:
        subst_len = random.randint(2, 8)
        subst = random.choices(lower_case_letters + math_symbols,
                               k=subst_len)
        substitute.update({char: subst})
        substitute_str.append(char)
        substitute_str.append(":")
        substitute_str.append("[")
        substitute_str.extend(subst)
        substitute_str.append("]")
        substitute_str.append(",")
    return substitute, substitute_str[:-1]


def gen_data(root, name, num, mode_str, modes="subst", vocab=None):

    with open(f'{root}/{mode_str}/{name}.src', 'w') as train_src, open(f'{root}/{mode_str}/{name}.tgt', 'w') as train_tgt:
        for _ in tqdm.tqdm(range(num)):
            new_vocab = random.sample(vocab, k=68)
            upper_case_letters = new_vocab[:24]
            lower_case_letters = new_vocab[24:48]
            math_symbols = new_vocab[48:68]
            mode = random.choice(modes)


            if mode in ["rewrite", "induct_rewrite"]:
                seq_len = random.randint(10, 20)
                lhs = random.choices(math_symbols, k=round(seq_len * 1.5)) + \
                      random.choices(lower_case_letters, k=seq_len)
                random.shuffle(lhs)
                lhs_pattern, init, substring_len = sample_substring(lhs, 5, 20)

                letters = []
                for v in lhs_pattern:
                    if v in lower_case_letters:
                        letters.append(v)
                letters = set(letters)
                upper_case_letters = upper_case_letters[:len(letters)]
                substitute = {c: l for c, l in zip(upper_case_letters, letters)}
                for char, letter in substitute.items():
                    lhs_pattern = Pattern(lhs_pattern).replace(letter, char)
                rhs_pattern_len = random.randint(5, 20)
                rhs_pattern = random.choices(math_symbols, k=rhs_pattern_len) + list(upper_case_letters)
                random.shuffle(rhs_pattern)
                rule = SubstRule(upper_case_letters, rhs_pattern)
                rhs_substring = rule.subst(substitute)
                rhs = lhs[:init] + rhs_substring + lhs[init + substring_len:]
                pattern = lhs_pattern + ["->"] + rhs_pattern

                train_src.write(" ".join([str(c) for c in ["<UPPER>"] + upper_case_letters])+ " ")
                train_src.write(" ".join([str(c) for c in ["<LOWER>"] + lower_case_letters])+ " ")
                train_src.write(" ".join([str(c) for c in ["<MATH>"] + math_symbols])+ " <space> ")
                if mode == "induct_rewrite":
                    train_src.write(" ".join([str(c) for c in lhs + ["<space>"] + rhs]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')
                elif mode == "rewrite":
                    train_src.write(" ".join([str(c) for c in pattern + ["<space>"] + lhs]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in rhs]) + '\n')


            elif mode in ["induct_hard", "induct_v2", "induct", "deduct", "abduct", "induct_v2_double", "induct_v3"]:
                num_chars = random.randint(3, 5)
                pattern_length = random.choice(range(5, 20))
                upper_case_letters = upper_case_letters[:num_chars]
                pattern_chars = random.choices(upper_case_letters, k=num_chars * 2)
                upper_case_letters = list(set(pattern_chars))
                pattern = pattern_chars+random.choices(math_symbols, k=pattern_length)
                random.shuffle(pattern)
                rule = SubstRule(upper_case_letters, pattern)
                substitute, substitute_str = gen_subst(upper_case_letters, lower_case_letters, math_symbols)
                result = rule.subst(substitute)
                substitute_2, substitute_str_2 = gen_subst(upper_case_letters, lower_case_letters, math_symbols)
                result_2 = rule.subst(substitute_2)

                train_src.write(" ".join([str(c) for c in ["<UPPER>"] + upper_case_letters]) + " ")
                train_src.write(" ".join([str(c) for c in ["<LOWER>"] + lower_case_letters]) + " ")
                train_src.write(" ".join([str(c) for c in ["<MATH>"] + math_symbols]) + " <space> ")

                if mode == "induct_hard":
                    train_src.write(" ".join([str(c) for c in result]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')
                elif mode == "induct_v2":
                    train_src.write(" ".join([str(c) for c in result]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in substitute_str + ["<space>"] + pattern]) + '\n')
                elif mode == "induct":
                    train_src.write(" ".join([str(c) for c in result + ["<space>"] + substitute_str]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')

                elif mode == "induct_v2_double":
                    train_src.write(" ".join([str(c) for c in result+["<space>"]+result_2]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in substitute_str + ["<space>"] +substitute_str_2 +["<space>"]+ pattern]) + '\n')
                elif mode == "induct_v3":
                    train_src.write(" ".join([str(c) for c in result+["<space>"]+result_2]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in pattern]) + '\n')
                elif mode == "deduct":
                    train_src.write(" ".join([str(c) for c in pattern+ ["<space>"] + substitute_str]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in result]) + '\n')
                elif mode == "abduct":
                    train_src.write(" ".join([str(c) for c in pattern+ ["<space>"] +result]) + '\n')
                    train_tgt.write(" ".join([str(c) for c in substitute_str]) + '\n')
            else:
                raise ValueError("Mode {} not found".format(mode))

def main(root, num_train, num_test, mode, vocab_size):
    mode_str = "-".join(mode)
    mode_str += "_vocab{}".format(vocab_size)
    mode_str += "_train{}M".format(num_train/1000000)
    try:
        shutil.rmtree(f'{root}/{mode_str}/')
    except:
        pass
    pathlib.Path(f'{root}/{mode_str}/').mkdir()
    vocab = list(range(vocab_size))

    generate(root=root, num_train=num_train, mode_str=mode_str,
             num_test=num_test, mode=mode, vocab=vocab)

def generate(root, num_train, mode_str, num_test, mode, vocab):
    root_bin = f'{root}/{mode_str}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    gen_data(root, 'train', num_train, mode_str, mode, vocab)
    gen_data(root, 'test', num_test, mode_str, mode, vocab)
    gen_data(root, 'valid', num_test, mode_str, mode, vocab)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'tgt', '--destdir', root_bin,
               '--trainpref', f'{root}/{mode_str}/train',
               '--validpref', f'{root}/{mode_str}/valid',
               '--testpref', f'{root}/{mode_str}/test',
               '--joined-dictionary'
               ]

    #subprocess.check_call(command)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        default="data", help="store directory")
    parser.add_argument("--mode", type=str, nargs='+',
                        default=["rewrite"], help="task mode")
    parser.add_argument("--num_train", type=int,
                        default=10000, help="num of train")
    parser.add_argument("--vocab_size", type=int,
                        default=100, help="num of train")
    parser.add_argument("--num_test",
                        type=int,
                        default=1000)
    args = parser.parse_args()

    main(args.root, args.num_train, args.num_test, args.mode, args.vocab_size)
