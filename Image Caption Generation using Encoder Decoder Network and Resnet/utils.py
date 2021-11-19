import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *
import re
import nltk
from nltk.tokenize import word_tokenize

from nltk.translate.bleu_score import sentence_bleu

def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of str): list of cleaned captions, words joined together as strings
    """
    image_ids = []
    cleaned_captions = []
    # QUESTION 1.1
    for line in lines:
        line=line.split("\t")
        img_id = line[0].split(".")[0]
        image_ids.append(img_id)
        clean_txt = re.sub(r'[^\w\s]', '', line[1]) 
        clean_txt =re.sub(r'[0-9]', '', clean_txt)
        cleaned_captions.append(clean_txt.lower().strip())
    return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with
    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words
    #Loop through clean captions to tokenize each sentence
    all_word_lists = [article.replace('\n',' ').split() for article in cleaned_captions]
    print(len(all_word_lists))
    print(all_word_lists[4])

    # Add all the tokens from each sentence into single list 
    all_vocab = []
    for word_list in all_word_lists:
        for word in word_list:
            all_vocab.append(word)
    print(len(all_vocab))

    #Loop through the list "all_vocab" and retain only the words that appear more than 3 times
    final_vocab = []
    for word in all_vocab:
        if all_vocab.count(word)>3:
            final_vocab.append(word) 
    print(len(final_vocab))
    
    #Retain only unique words in list "final_vocab" and add the unique words to list uniq_final_vocab
    uniq_final_vocab = set(final_vocab)
    print(len(uniq_final_vocab))

    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    # vocab.add_word('word')
    final_vocab = [vocab.add_word(word) for word in uniq_final_vocab]

    return vocab

"""
def decode_caption(sampled_ids, vocab):
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    predicted_caption = ""

    # QUESTION 2.1
    return predicted_caption
"""  
    
### This method  converts the predicted caption to readable format and adds the prediction to the dictionary "all_ref_pred" with the key "predicted" against a image_id
def decode_caption( sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    #print(sentence)
    # QUESTION 2.1     
    sentence = sampled_ids
    pred_final_sentence = ""
    for word_id in sentence:
        word = vocab.idx2word[word_id]
        word = word.strip()
        word = word.lstrip(' ')
        #print("--"+word +"--")
        if word == '<end>':  
            break
        if(word=="<start>" or word=="<unk>"):
            continue
        #print("word--", word)
        #sampled_caption.append(word)
        pred_final_sentence = pred_final_sentence+ " " +word  
    pred_final_sentence =  word_tokenize(pred_final_sentence)
    #print(pred_final_sentence)
    return pred_final_sentence

"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions,image_ids,sentence = zip(*data)
    #print("util len captions:", len(captions))#63
    #print("util img len:", len(images))#64
    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 
    #print("util img type:", images.shape) #64,2048
    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions] # [ 23, 4, 6,9... till 64]
    #print("util lengths:", lengths) #64,2048
    targets = torch.zeros(len(captions), max(lengths)).long()
    #print("util targets type:", targets.shape) # 64, max length in length
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]  
    #print(targets[63])        
    return images, targets, lengths,image_ids, sentence


### Code to construct json file that holds actual test captions & predected captions after readig test file. Single JSON file that has actual caption & predicted captions
def create_reference_predicted_json():
    lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, cleaned_captions = parse_lines(lines)
    all_ref_pred = {}
    z=0
    for k,sent in enumerate(cleaned_captions):
    #     if(z==10):
    #         break
        z=z+1
        img_id = test_image_ids[k]
        sent = sent.split(" ")
        if img_id in all_ref_pred:                
            inner_json = all_ref_pred[img_id]
            #print(inner_json)
            list1 = inner_json["actualcaptions"]
            list1.append(sent)
            inner_json.update({'actualcaptions': list1})
            all_ref_pred[img_id] = inner_json
        else:
            inner_json = {}
            list1 = []
            list1.append(sent)
            inner_json.update({'actualcaptions': list1})
            all_ref_pred[img_id] = inner_json
        #print((all_ref_pred))
    #print(all_ref_pred["377872672_d499aae449"]["original"])
    return all_ref_pred