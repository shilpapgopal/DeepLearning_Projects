"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image


from datasets import Flickr8k_Images, Flickr8k_Features, Flickr8k_Test
from models import DecoderRNN, EncoderCNN
from utils import *
import utils
from config import *
import json

import nltk
from nltk.tokenize import word_tokenize

from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import SmoothingFunction

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = True


# reconstruct the captions and vocab, just as in extract_features.py
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)

#vocab = build_vocab(cleaned_captions) # shilpa commented

#=========BEGIN Code to load & biuilt vocab=========================================
# As building vocab with unique words takes lot of time because of "set" functionality, once the vocab was built with 3.2K words, the words were written to a text file 
# and the words are loaded to vocab instead of building the vocab each time. 
with open("uniqwordlist.txt", "r") as filestream:
    for line in filestream:
        vocab_words=line.split(",")
print(len(vocab_words))

vocab = Vocabulary()
# add the token words
vocab.add_word('<pad>')
vocab.add_word('<start>')
vocab.add_word('<end>')
vocab.add_word('<unk>')

# TODO add the rest of the words from the cleaned captions here
# vocab.add_word('word')
final_vocab = [vocab.add_word(word) for word in vocab_words]
print("vocab len-",vocab.__len__())
#=========END Code to load & biuilt vocab=========================================


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

# initialize the models and set the learning parameters
decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


if not EVAL:

    # load the features saved from extract_features.py
    features = torch.load('features.pt', map_location=device)
    print("Loaded features:", features.shape)

    features = features.repeat_interleave(5, 0)
    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64, # change as needed
        shuffle=True,
        num_workers=2, # may need to set to 0
        collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
    )


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)



#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
# 
#########################################################################

    # TODO write training loop on decoder here


    # for each batch, prepare the targets using this function in utils.py
    # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

    num_epoc=5
    print_at = 100
    total_step = len(train_loader)
    epoch_loss=0
    for epoch in range(num_epoc):   
        epoch_loss = 0       
        for i, (feature, captions, lengths) in enumerate(train_loader):
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            optimizer.zero_grad()
            outputs = decoder(feature, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % print_at == 0:
                print('Epoch [{}/{}], print_at [{}/{}], Loss: {:.3f}'.format(epoch, num_epoc, i, total_step, loss.item())) 

    # save model after training
    decoder_ckpt = torch.save(decoder, "decoder.ckpt")

# if we already trained, and EVAL == True, reload saved model
else:

    data_transform = transforms.Compose([ 
        transforms.Resize(224),     
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])


    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)

    # load models
    encoder = EncoderCNN().to(device)
    decoder = torch.load("decoder.ckpt").to(device)
    encoder.eval()
    decoder.eval() # generate caption, eval mode to not influence batchnorm

    dataset_test = Flickr8k_Test(
        image_ids=test_image_ids,
        captions=test_cleaned_captions,
        transform=data_transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    print(len(dataset_test))
    print(len(test_loader))
    print(len(test_image_ids))
    

#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
# 
#########################################################################

    # TODO define decode_caption() function in utils.py
    # predicted_caption = decode_caption(word_ids, vocab)    
    #def decode_caption(self, word_ids, vocab):
    #    return predicted_caption
    
    
    #------------ Newly defined method in util to create a json that holds actual and predicted captions against a imageid for whole testdata set ------------
    all_ref_pred = create_reference_predicted_json()
    
    #------------ Below block of code does the prediction for the whole test dataset by called "decode_caption" method in util.py and stores the predction for each image in a json file called "all_ref_pred". Late this JSON is used to calculate BLEU score and Cosine similarity score on actual vs predicted captions ------------
    z=0
    for i,(test_image_original,image_ids,original_sentence) in enumerate(test_loader):
        z=z+1
        #if(z==3):
        #  break
        test_image_original = test_image_original.to(device)
        features = encoder(test_image_original)
        prediction = decoder.sample(features)

        #loop through the predictions array and decode prediction for each image
        for i in range(prediction.shape[0]):        
            sentence = prediction[i].cpu().numpy()
            
            # call decode_caption function to get the redable format of prediction
            pred_final_sentence = decode_caption(sentence,vocab)
    
            # add the predicted sentence in all_ref_pred json so that we later can we during BLEU scoring
            if image_ids[i] in all_ref_pred: 
                if('predicted' in all_ref_pred[image_ids[i]]):
                    inner_json = all_ref_pred[image_ids[i]]
                    list1 = inner_json["predicted"]
                    list1.append(pred_final_sentence)
                    inner_json.update({'predicted': list1})
                    all_ref_pred[image_ids[i]] = inner_json
                else:
                    inner_json = all_ref_pred[image_ids[i]]
                    list1 = []
                    list1.append(pred_final_sentence)
                    inner_json.update({'predicted': list1})
                    all_ref_pred[image_ids[i]] = inner_json       
    

#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity 
# 
#########################################################################


    # Feel free to add helper functions to utils.py as needed,
    # documenting what they do in the code and in your report

    ### Write the JSON that has actual caption & predected caption to a text file. Instead of predicting each time again n again, this JSON file is used for BLEU & Cosine scoring
    f = open("prediction_for_BLEU_score.txt", "w")
    f.write(json.dumps(all_ref_pred))
    f.close()
    
    
    ### Before BLEU scoring import the JSON which has actual captions & predicted captions 
    with open('prediction_for_BLEU_score.txt') as json_file:
        all_ref_pred = json.load(json_file)
    
    
    ### This method calculates the BLEU score.
    ### Use the JSON file created that has actual and predicted captions to calculate average blue score.
    ### Here 0.25 weight is used for each of 1-gram, 2-gram, 3-gram & 4-grams, smoothing function used is method4 
    def calculate_bleu_score():
        print("-----------BEGIN BLEU SCORING----------------")
        image_count=0
        cumulative_score=0
        s1=0
        cc = SmoothingFunction()
        for image in all_ref_pred:
            if("predicted" in all_ref_pred[image]):
                image_count=image_count+1
                reference = all_ref_pred[image]["actualcaptions"]
                candidate = all_ref_pred[image]["predicted"][0]

                cumulative_score+=sentence_bleu(reference,candidate,weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4)
                
                # calc BLEU score using 0.25 weight for each gram & smoothing= method4
                s1=sentence_bleu(reference,candidate,weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4)
                
                bleu_min =0 # Min Bleu score noted
                bleu_max=1.6736959844303068# Max Bleu score noted
                s1 = (s1 - bleu_min) / (bleu_max - bleu_min)
                if(s1>0.7):
                    print(image, "---HIGH BLEU SCORE---", s1 , "--actual--",reference[0], "--pred--",candidate)

                if(s1<0.08):
                    print(image, "---LOW BLEU SCORE---", s1 , "--actual--",reference[0], "--pred--",candidate)
                 
        print( "Average score is: " , cumulative_score/image_count)
        print("image_count:" ,image_count)
        
        
    ### Invoke the method to print the average/high and low bleu score on the predictions    
    calculate_bleu_score()    
    
    
    ### This method finds the mean vector for each sentence. 
    ### The input passed to this method is a sentence which can be actual or predicted caption.
    ### The id of each word in sentence is found and then embedding for each work is retreived.
    ### Then the mean vector of all the words is calculated using torch.mean
    ### The function returns a tensor of size 1,256
    def find_sentence_mean(sentence):
        embed_tensor = torch.tensor(())
        for word in sentence:
            try:
                idx = torch.tensor([vocab.word2idx[word]], dtype =torch.long)
                embedding = decoder.embed(idx)
                #print("embed_tensor shape:", embedding.shape) (1,256)
                embed_tensor = torch.cat([embed_tensor, embedding],0)
                #print("embed_tensor shape:",embed_tensor.shape) # (7,256) keeps adding
            except:
                pass        
        #print("---embed_tensor shape:",embed_tensor.shape) #torch.Size([7, 256])
        #print("torch mean---",torch.mean(embed_tensor, 0).shape) # torch.Size([256])
        embed_tensor = torch.unsqueeze(torch.mean(embed_tensor, 0),0) #torch.Size([1, 256])
        return embed_tensor    
    
    ### This method prints the average/high & low Cosine score on predicted image captions
    def calculate_cosine_score():
        print("-----------BEGIN COSINE SCORING----------------")
        count_image=0
        count_score_match=0
        for image in all_ref_pred:
        #     if(count_image==10):
        #         break
            if("predicted" in all_ref_pred[image]):
                count_image=count_image+1
                actual = all_ref_pred[image]["actualcaptions"]
                pred_sent = all_ref_pred[image]["predicted"][0]
                pred_mean=find_sentence_mean(pred_sent)
                list_all_cos_sim = []
                for ref_sent in actual:
                    ref_mean = find_sentence_mean(ref_sent)
                    cosine_sim_value = cosine_similarity(pred_mean.detach().numpy() ,ref_mean.detach().numpy() )
                    #print(cosine_sim_value)
                    list_all_cos_sim.append(cosine_sim_value)

                avg = sum(list_all_cos_sim) / 5
                if(avg[0][0] >0.7 or avg[0][0] <0.1):
                    print(avg)
                    #print(list_all_cos_sim)
                    count_score_match=count_score_match+1
                    print("Img:",image,"--", avg[0][0])
                    print("pred sent: " , pred_sent)
                    print("ref_sent:", ref_sent)
                #print("avg - ", sum(list_all_cos_sim) / 5) 
        print("Count:",count_score_match)        
        
    ### Invoke the method to print the average/high and low Cosine score on the predictions    
    calculate_cosine_score()
