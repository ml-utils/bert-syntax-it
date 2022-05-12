# -*- coding: utf-8 -*-
"""notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PKYTH2q2bncPfBxLnZg5LQ7fCbklBkCc
"""

#!git clone https://github.com/ml-utils/bert-syntax-it.git

#!pip install folium==0.2.1
#!pip install pytorch-pretrained-bert

import os.path
from collections import Counter
import json
import argparse, sys
import csv

import torch
from pytorch_pretrained_bert import BertForMaskedLM,tokenization
from torch.nn.functional import softmax
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

import bert_utils
from bert_utils import load_testset_data, analize_sentence, get_probs_for_words, tokenize_sentence, \
    estimate_sentence_probability_from_text


def load_it():
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
    out = []
    for line in open("it_dataset.tsv"):
        case = line.strip().split("\t")
        cc[case[1]]+=1
        g,ug = case[-2],case[-1]
        g = g.split()
        ug = ug.split()
        assert(len(g)==len(ug)),(g,ug)
        diffs = [i for i,pair in enumerate(zip(g,ug)) if pair[0]!=pair[1]]
        if (len(diffs)!=1):
            #print(diffs)
            #print(g,ug)
            continue
        assert(len(diffs)==1),diffs
        gv=g[diffs[0]]   # good
        ugv=ug[diffs[0]] # bad
        g[diffs[0]]="***mask***"
        g.append(".")
        out.append((case[0],case[1]," ".join(g),gv,ugv))
    return out

def load_marvin():
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
    out = []
    for line in open("marvin_linzen_dataset.tsv"):
        case = line.strip().split("\t")
        cc[case[1]]+=1
        g,ug = case[-2],case[-1]
        g = g.split()
        ug = ug.split()
        assert(len(g)==len(ug)),(g,ug)
        diffs = [i for i,pair in enumerate(zip(g,ug)) if pair[0]!=pair[1]]
        if (len(diffs)!=1):
            #print(diffs)
            #print(g,ug)
            continue    
        assert(len(diffs)==1),diffs
        gv=g[diffs[0]]   # good
        ugv=ug[diffs[0]] # bad
        g[diffs[0]]="***mask***"
        g.append(".")
        out.append((case[0],case[1]," ".join(g),gv,ugv))
    return out


def eval_it(bert,tokenizer):
    o = load_it()
    print(len(o), file=sys.stderr)
    from collections import defaultdict
    import time
    rc = defaultdict(Counter)
    tc = Counter()
    start = time.time()
    bert_utils.print_orange(f'{len(o)} sentences to process..')
    for i, (case, tp, s, good_word, bad_word) in enumerate(o):
        ps = get_probs_for_words(bert, tokenizer, s, good_word, bad_word)
        if ps is None: ps = [0, 1]
        gp = ps[0]
        bp = ps[1]
        print(gp > bp, case, tp, good_word, bad_word, s)
        if i % 100 == 0:
            print(f'{bert_utils.bcolors.WARNING}{i}{bert_utils.bcolors.ENDC}')
            print(i, time.time() - start, file=sys.stderr)
            start = time.time()
            sys.stdout.flush()

def eval_marvin(bert,tokenizer):
    o = load_marvin()
    print(len(o),file=sys.stderr)
    from collections import defaultdict
    import time
    rc = defaultdict(Counter)
    tc = Counter()
    start = time.time()
    for i,(case,tp,s,g,b) in enumerate(o):
        ps = get_probs_for_words(bert,tokenizer,s,g,b)
        if ps is None: ps = [0,1]
        gp = ps[0]
        bp = ps[1]
        print(gp>bp,case,tp,g,b,s)
        if i % 100==0:
            print(i,time.time()-start,file=sys.stderr)
            start=time.time()
            sys.stdout.flush()

def eval_lgd(bert,tokenizer):
    for i,line in enumerate(open("lgd_dataset_with_is_are.tsv",encoding="utf8")):
        na,_,masked,good,bad = line.strip().split("\t")
        ps = get_probs_for_words(bert,tokenizer,masked,good,bad)
        if ps is None: continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp>bp),na,good,gp,bad,bp,masked.encode("utf8"),sep=u"\t")
        if i%100 == 0:
            print(i,file=sys.stderr)
            sys.stdout.flush()


def read_gulordava():
    rows = csv.DictReader(open("generated.tab",encoding="utf8"),delimiter="\t")
    data=[]
    for row in rows:
        row2=next(rows)
        assert(row['sent']==row2['sent'])
        assert(row['class']=='correct')
        assert(row2['class']=='wrong')
        sent = row['sent'].lower().split()[:-1] # dump the <eos> token.
        good_form = row['form']
        bad_form  = row2['form']
        sent[int(row['len_prefix'])]="***mask***"
        sent = " ".join(sent)
        data.append((sent,row['n_attr'],good_form,bad_form))
    return data


def eval_gulordava(bert,tokenizer):
    for i,(masked,natt,good,bad) in enumerate(read_gulordava()):
        if good in ["is","are"]:
            print("skipping is/are")
            continue
        ps = get_probs_for_words(bert,tokenizer,masked,good,bad)
        if ps is None: continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp>bp),natt,good,gp,bad,bp,masked.encode("utf8"),sep=u"\t")
        if i%100 == 0:
            print(i,file=sys.stderr)
            sys.stdout.flush()

# choose_eval()


def init_bert_model(model_name, dict_name=None, do_lower_case=False) -> (BertPreTrainedModel, BertTokenizer):
    # model_name = 'bert-large-uncased'
    #if 'base' in sys.argv: model_name = 'bert-base-uncased'
    print(f'model_name: {model_name}')
    print("loading model:", model_name, file=sys.stderr)
    bert = BertForMaskedLM.from_pretrained(model_name)
    print("bert model loaded, getting the tokenizer..")

    if dict_name is None:
        vocab_filepath = model_name
    else:
        vocab_filepath = os.path.join(model_name, 'dict.txt')
    tokenizer = tokenization.BertTokenizer.from_pretrained(vocab_filepath, do_lower_case=do_lower_case)

    print("tokenizer ready.")

    bert.eval()
    return bert, tokenizer


def run_eval(eval_suite, bert: BertPreTrainedModel, tokenizer: BertTokenizer):
    print(f'running eval, eval_suite: {eval_suite}')
    if 'it' == eval_suite:
        eval_it(bert, tokenizer)
    elif 'marvin' == eval_suite:
        eval_marvin(bert,tokenizer)
    elif 'gul' == eval_suite:
        eval_gulordava(bert,tokenizer)
    else:
        eval_lgd(bert,tokenizer)


def arg_parse():
    print('parsing args..')
    # Python program to demonstrate
    # command line arguments

    import getopt, sys

    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    options = "be:"

    # Long options
    long_options = ["bert_model", "eval_suite"]

    DEFAULT_MODEL = 'bert-large-uncased'
    DEFAULT_EVAL_SUITE = 'lgd'
    model_name = DEFAULT_MODEL
    eval_suite = DEFAULT_EVAL_SUITE

    try:
        # Parsing argument
        print(f'argumentList: {argumentList}')

        # checking each argument
        for arg_idx, currentArgument  in enumerate(argumentList):
            print(f'persing currentArgument {currentArgument}')
            if currentArgument in ("-h", "--Help"):
                print("Displaying Help")

            elif currentArgument in ("-b", "--bert_model"):

                argValue = argumentList[arg_idx+1]
                print(f'currentArgument: {currentArgument}, argValue: {argValue}')
                if argValue == 'base':
                    model_name = 'bert-base-uncased'
                else:
                    model_name = argValue
                    print(f'set model_name: {model_name}')

            elif currentArgument in ("-e", "--eval_suite"):
                argValue = argumentList[arg_idx + 1]
                print(f'currentArgument: {currentArgument}, argValue: {argValue}')
                eval_suite = argValue

    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))

    print(f'model_name {model_name}, eval_suite {eval_suite}')
    return model_name, eval_suite


import numpy as np
from scipy.special import softmax



def get_masked_word_probability(bert, tokenizer):
    return 0


def custom_eval(sentence, bert, tokenizer):
    bert, tokenizer = init_bert_model('bert-base-uncased')

    compare_tokens, compare_target_idx = tokenize_sentence(tokenizer, "What is ***your*** name?")

    bare_sentence_tokens = tokenizer.tokenize(sentence)

    paper_logprobs = bert_get_logprobs(bare_sentence_tokens, None, bert, tokenizer, device=None)

    tokens_list = ['[CLS]'] + bare_sentence_tokens + ['[SEP]']
    target_idx = 3
    masked_word = tokens_list[target_idx]
    tokens_list[target_idx] = '[MASK]'

    print(f'tokens: {tokens_list}, masked_word: {masked_word}')
    print(f'compare_tokens: {compare_tokens}')

    input_ids = tokenizer.convert_tokens_to_ids(tokens_list)

    try:
        masked_word_id = tokenizer.convert_tokens_to_ids([masked_word])
    except KeyError:
        print(f"unable to convert {masked_word} to id")
        return None
    tens = torch.LongTensor(input_ids).unsqueeze(0)

    res_unsliced = bert(tens)
    res=res_unsliced[0, target_idx]

    # res=torch.nn.functional.softmax(res,-1)

    pred = bert("What is [MASK] name?")

    # Set the maximum sequence length.
    MAX_LEN = 128
    # Pad our input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


def print_sentence_pairs_probabilities(bert: BertPreTrainedModel, tokenizer: BertTokenizer, sentence_data):
    sentence_good_no_extraction = sentence_data['sentence_good_no_extraction']
    sentence_bad_extraction = sentence_data['sentence_bad_extraction']
    sentence_good_extraction_resumption = sentence_data['sentence_good_extraction_resumption']
    sentence_good_extraction_as_subject = sentence_data['sentence_good_extraction_as_subject']
    print(f'sentence_good_no_extraction: {sentence_good_no_extraction}')
    print(f'sentence_bad_extraction: {sentence_bad_extraction}')
    print(f'sentence_good_extraction_resumption: {sentence_good_extraction_resumption}')
    print(f'sentence_good_extraction_as_subject: {sentence_good_extraction_as_subject}')

    prob_sentence_good_no_extraction = estimate_sentence_probability_from_text(bert, tokenizer, sentence_good_no_extraction)
    prob_sentence_bad_extraction = estimate_sentence_probability_from_text(bert, tokenizer, sentence_bad_extraction)
    prob_sentence_good_extraction_resumption = estimate_sentence_probability_from_text(bert, tokenizer, sentence_good_extraction_resumption)
    prob_sentence_good_extraction_as_subject = estimate_sentence_probability_from_text(bert, tokenizer, sentence_good_extraction_as_subject)

    print(f'prob_sentence_good_no_extraction: {prob_sentence_good_no_extraction}')
    print(f'prob_sentence_bad_extraction: {prob_sentence_bad_extraction}')
    print(f'prob_sentence_good_extraction_resumption: {prob_sentence_good_extraction_resumption}')
    print(f'prob_sentence_good_extraction_as_subject: {prob_sentence_good_extraction_as_subject}')


def main():
    print('main')

    model_name, eval_suite = arg_parse()
    model_name = 'bert-base-uncased'  # NB bert large uncased is about 1GB
    model_name = f'''models/bert-base-italian-uncased/'''
    model_name = f'''models/bert-base-italian-cased/'''
    model_name = f'models/bert-base-italian-xxl-cased/'
    # model_name = f'./models/gilberto-uncased-from-camembert.tar.gz'
    eval_suite = 'it'
    bert, tokenizer = init_bert_model(model_name, do_lower_case=False)
    if tokenizer is None:
        print('error, tokenizer is null')
        return

    #
    bert_utils.check_unknown_words(tokenizer)
    sentence_to_analizse = 'Di che cosa Marco si chiede se è stata riparata da ***Luca***?'
    topk_tokens, topk_probs, topk_probs_nonsoftmax = analize_sentence(bert, tokenizer, sentence_to_analizse)
    print(f'sentence: {sentence_to_analizse}')
    print(f'topk: {topk_tokens}, top_probs: {topk_probs}, topk_probs_nonsoftmax: {topk_probs_nonsoftmax}')

    testsets_dir = './outputs/syntactic_tests_it/'
    testset_files = [#'variations_tests.jsonl',
                     'wh_adjunct_islands.jsonl', 'wh_complex_np_islands.jsonl', 'wh_subject_islands.jsonl',
                     'wh_whether_island.jsonl'
                     ]
    for test_file in testset_files:
        run_testset(testsets_dir, test_file, bert, tokenizer)

    # run_eval(eval_suite, bert, tokenizer)
    #prob1 = estimate_sentence_probability_from_text(bert, tokenizer, 'What is your name?')
    #prob2 = estimate_sentence_probability_from_text(bert, tokenizer, 'What is name your?')
    #print(f'prob1: {prob1}, prob2: {prob2}')
    #eval_it(bert, tokenizer)
    #custom_eval("What is your name?", bert, tokenizer)


def run_testset(testsets_dir: str, filename: str, bert: BertPreTrainedModel, tokenizer: BertTokenizer):
    filepath = os.path.join(testsets_dir, filename)
    bert_utils.print_orange(f'running test {filepath}')
    testset_data = bert_utils.load_testset_data(filepath)
    examples_count = len(testset_data['sentences'])
    print(f'examples_count: {examples_count}')

    # todo: add checks that there is enough variability / less repetitive examples (subjects proper names or pronouns,
    #  plural and singular, 1st, 2nd and 3rd person, ..

    # only_examples = [3, 6, 8, 10, 14, 15, 16, 18, 19, 21, 22, 23, 26, 29, 31, 32, 33, 39, 43, 46, 47, 48, 49]
    # print(f'incorrect examples count: {len(only_examples)} out of 50 ({len(only_examples)/50})')
    error_count_base_sentence = 0
    error_count_second_sentence = 0
    error_count_either = 0
    no_errors_examples_indexes = []
    examples_by_base_sentence_acceptability_diff = {}
    examples_by_second_sentence_acceptability_diff = {}
    second_sentences_by_penLP = {}
    bad_sentences_by_penLP = {}
    base_sentences_by_penLP = {}
    for example_idx, sentence_data in enumerate(testset_data['sentences']):
        #print(f"json_str, type: {type(sentence_data)}: {sentence_data}")
        # print_sentence_pairs_probabilities(bert, tokenizer, sentence_data)
        base_sentence_less_acceptable, second_sentence_less_acceptable, \
        acceptability_diff_base_sentence, acceptability_diff_second_sentence, \
        penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence, \
        logitis_normalized_bad_sentence, logitis_normalized_base_sentence, logitis_normalized_2nd_good_sentence, \
        oov_counts \
            = bert_utils.analize_example(bert, tokenizer, example_idx, sentence_data)
        #return
        sentences = bert_utils.get_sentences_from_example(sentence_data)

        second_sentences_by_penLP[penLP_2nd_good_sentence] = sentences[2]
        bad_sentences_by_penLP[penLP_bad_sentence] = sentences[1]
        base_sentences_by_penLP[penLP_base_sentence] = sentences[0]
        examples_by_base_sentence_acceptability_diff[acceptability_diff_base_sentence] \
            = example_as_tuple(example_idx, penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence,
               oov_counts, sentences[0], sentences[1])
        examples_by_second_sentence_acceptability_diff[acceptability_diff_second_sentence] \
            = example_as_tuple(example_idx, penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence,
               oov_counts, sentences[2], sentences[1])

        if base_sentence_less_acceptable:
            error_count_base_sentence += 1
        if second_sentence_less_acceptable:
            error_count_second_sentence += 1

        if base_sentence_less_acceptable or second_sentence_less_acceptable:
            error_count_either += 1
        else:
            no_errors_examples_indexes.append(example_idx)

    print(f'error count and accuracy rates from {examples_count} examples: '
          f'base sentence {error_count_base_sentence} '
          f'(acc: {get_perc(examples_count-error_count_base_sentence, examples_count)}), '
          f'second sentence: {error_count_second_sentence} '
          f'(acc: {get_perc(examples_count-error_count_second_sentence, examples_count)}), '
          f'either: {error_count_either} '
          f'(acc: {get_perc(examples_count-error_count_either, examples_count)}), '
          f'filename: {filename}')

    print(f'error count out of {examples_count} examples: base sentence {error_count_base_sentence} '
          f'({get_perc(error_count_base_sentence, examples_count)}), second sentence: {error_count_second_sentence} '
          f'({get_perc(error_count_second_sentence, examples_count)}), either: {error_count_either} '
          f'({get_perc(error_count_either, examples_count)}), filename: {filename}')

    # print examples getting no errors:
    bert_utils.print_orange('Examples getting no errors:')
    for example_idx in no_errors_examples_indexes:
        no_error_example = testset_data['sentences'][example_idx]
        print(f'{bert_utils.get_sentences_from_example(no_error_example)}')

    bert_utils.print_orange('examples sorted by sentence_acceptability diff, second sentence:')
    for acceprability_diff, example in dict(sorted(examples_by_second_sentence_acceptability_diff.items())).items():
        print_example(example, acceprability_diff, compare_with_base_sentence=False)

    bert_utils.print_orange('examples sorted by sentence_acceptability diff, base sentence:')
    for acceprability_diff, example in dict(sorted(examples_by_base_sentence_acceptability_diff.items())).items():
        print_example(example, acceprability_diff, compare_with_base_sentence=True)

    print_sentences_sorted_by_PenLP(second_sentences_by_penLP, 'second sentences sorted by PenLP:')
    print_sentences_sorted_by_PenLP(bad_sentences_by_penLP, 'bad sentences sorted by PenLP:')
    print_sentences_sorted_by_PenLP(base_sentences_by_penLP, 'base sentences sorted by PenLP:')


def print_sentences_sorted_by_PenLP(sentences_by_penLP, msg):
    bert_utils.print_orange(msg)
    for PenLP, sentence in dict(sorted(sentences_by_penLP.items())).items():
        print(f'{PenLP:.1f} {sentence}')


def example_as_tuple(example_idx, penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence,
               oov_counts, sentence_good, sentence_bad):
    return (example_idx, penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence,
               oov_counts, sentence_good, sentence_bad)


def print_example(example, acceprability_diff, compare_with_base_sentence = True):
    example_idx = example[0]
    penLP_base_sentence = example[1]
    penLP_bad_sentence = example[2]
    penLP_2nd_good_sentence = example[3]
    oov_counts = example[4]
    sentence_good = example[5]
    sentence_bad = example[6]
    if compare_with_base_sentence:
        diff_descr = 'accept_diff_w_base_sent'
    else:
        diff_descr = 'accept_diff_w_2nd_sent'

    print(f'{diff_descr}: {rnd(acceprability_diff,3)}, '
          f'(PenLP values: {rnd(penLP_base_sentence,1)}, {rnd(penLP_bad_sentence,1)}, {rnd(penLP_2nd_good_sentence,1)}), '
          f'example (oov_counts: {oov_counts}): ({example_idx}, \'{sentence_good}\', \'{sentence_bad}\'')


def rnd(num, decimal_places):
    if num is not None:
        return round(num, decimal_places)
    else:
        return None


def get_perc(value, total):
    perc = (value / total) * 100
    return f'{perc:.1f} %'


def interactive_mode():
    print(f'interactive mode')

    # load model than wait for input sentences
    model_name = f'models/bert-base-italian-xxl-cased/'
    eval_suite = 'it'
    bert, tokenizer = init_bert_model(model_name, do_lower_case=False)

    print(f'model loaded, waiting for sentences..')

    # given two sentences, print PenLPs, and diff btw PenLPs
    end_program = False
    while not end_program:
        good_sentence = input('Enter first sentence (good): ')
        if good_sentence == 'exit':
            return
        bad_sentence = input('Enter 2nd sentence (bad): ')

        example = {'good_sentence': good_sentence, 'bad_sentence': bad_sentence, 'good_sentence2': None}

        base_sentence_less_acceptable, second_sentence_less_acceptable, \
        acceptability_diff_base_sentence, acceptability_diff_second_sentence, \
        penLP_base_sentence, penLP_bad_sentence, penLP_2nd_good_sentence, \
        logitis_normalized_bad_sentence, logitis_normalized_base_sentence, logitis_normalized_2nd_good_sentence, \
        oov_counts \
            = bert_utils.analize_example(bert, tokenizer, -1, example)
        diff_penLP = round(penLP_base_sentence - penLP_bad_sentence, 3)

        bert_utils.print_red(f'PenLP:')
        print(f'Diff {bert_utils.red_txt(diff_penLP)}, '
              f'good ({penLP_base_sentence:.1f}), bad ({penLP_bad_sentence:.1f}): {good_sentence} || {bad_sentence}')

        # analize both sentences with topk for each masking
        if diff_penLP >= 0:
            print_detailed_sentence_info(bert, tokenizer, good_sentence)
            print_detailed_sentence_info(bert, tokenizer, bad_sentence)


def print_detailed_sentence_info(bert, tokenizer, sentence_txt):
    bert_utils.print_red(f'printing details for sentence {sentence_txt}')
    tokens = tokenizer.tokenize(sentence_txt)
    sentence_ids = tokenizer.convert_tokens_to_ids(tokens)
    bert_utils.estimate_sentence_probability(bert, tokenizer, sentence_ids, verbose=True)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        interactive_mode()
    else:
        print(f'running main function')
        main()


