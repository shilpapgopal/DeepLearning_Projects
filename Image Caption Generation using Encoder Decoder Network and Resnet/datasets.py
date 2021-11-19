import torch
from PIL import Image
from torch.utils.data import Dataset

from config import IMAGE_DIR



class Flickr8k_Images(Dataset):
    """ Flickr8k custom dataset to read image data only,
        compatible with torch.utils.data.DataLoader. """
    
    def __init__(self, image_ids, transform=None):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            image_ids (str list): list of image ids
            transform: image transformer
        """
        self.image_ids = image_ids
        self.transform = transform


    def __getitem__(self, index):
        """ Returns image. """

        image_id = self.image_ids[index]
        path = IMAGE_DIR + str(image_id) + ".jpg"
        #print("path----",path)
        image = Image.open(open(path, 'rb'))

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_ids)


class Flickr8k_Features(Dataset):
    """ Flickr8k custom dataset with features and vocab, compatible with torch.utils.data.DataLoader. """
    
    def __init__(self, image_ids, captions, vocab, features):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            image_ids (str list): list of image ids
            captions (str list): list of str captions
            vocab: vocabulary wrapper
            features: torch Tensor of extracted features
        """
        #print("in init of flicker features--")
        self.image_ids = image_ids
        self.captions = captions
        self.vocab = vocab
        self.features = features


    def __getitem__(self, index):
        """ Returns one data pair (feature and target caption). """
        #print("--in get---")
        path = IMAGE_DIR + str(self.image_ids[index]) + ".jpg"
        img_id = self.image_ids[index]
        image_features = self.features[index]
        #print("dataset image_ids---",self.image_ids)
        # convert caption (string) to word ids.
        #print("---dataset self.captions[index] -", self.captions[index])
        tokens = self.captions[index].split()

        #print("---dataset len -", len(tokens))
        #print("---dataset tokens -", tokens)
        #print("---dataset vocab -", self.vocab)
        caption = []
        # build the Tensor version of the caption, with token words
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        #print("---dataset caption -", caption)
        target = torch.Tensor(caption)
        #print("---dataset target len -", target.shape)
        #print("image_features len -", image_features.shape)
        return image_features, target,img_id,self.captions[index]

    def __len__(self):
        #print("hellooooooooooooo")
        return len(self.image_ids)
        
class Flickr8k_Test(Dataset):
    """ Flickr8k custom dataset , compatible with torch.utils.data.DataLoader. """
    
    def __init__(self, image_ids, captions, transform):
        """ Set the path for images, captions and vocabulary wrapper.
        
        Args:
            image_ids (str list): list of image ids
            captions (str list): list of str captions
            transform: torch Tensor of extracted features
        """
        #print("in init of flicker features--")
        self.image_ids = image_ids
        self.captions = captions
        self.transform = transform

    def __getitem__(self, index):
        """ Returns one data pair (feature and target caption). """
        #print("--in get---")
        path = IMAGE_DIR + str(self.image_ids[index]) + ".jpg"
        img_id = self.image_ids[index]
        
        #--shilpa
        image_original = Image.open(open(path, 'rb'))

        if self.transform is not None:
            image_original = self.transform(image_original)
        #shil
        #print("---dataset self.captions[index] -", self.captions[index])
 
        ref_sentence = self.captions[index]

        return image_original, img_id,ref_sentence

    def __len__(self):
        #print("hellooooooooooooo")
        return len(self.image_ids)
        