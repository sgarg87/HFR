import hash_sentences
import argparse
import numpy as np


def read_sentences(file_path):
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    sentences = np.array(sentences)
    return sentences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cores", default=1, help="number of cores", type=int)
    parser.add_argument("-fp", "--file_path", default=1, help="file path for text sentences")
    args = parser.parse_args()
    del parser

    print('num_cores', args.cores)
    print('file_path', args.file_path)

    sentences = read_sentences(file_path=args.file_path)

    hash_sentences_obj = hash_sentences.HashSentences(
        hash_func='RkNN',
        num_cores=args.cores,
        is_wordvec_large=False,
        alpha=128,
        is_bert_embedding=False,
        is_zero_kernel_compute_outside_cluster=False,
    )

    sentence_hashcodes, _, _, _ = hash_sentences_obj.main(
        sentences=sentences,
        is_remove_pos_tags=True,
        num_hash_bits=30,
        is_self_dialog_learning=False,
        seed_val=0,
    )
    assert sentence_hashcodes.shape[0] == sentences.size
    np.save('sentences_hashcodes', sentence_hashcodes)
