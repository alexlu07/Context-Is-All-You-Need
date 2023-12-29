from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def twitter_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor

def alternate_preprocessor():
    preprocessor = TextPreProcessor(
        normalize=['url', 'email', 'phone', 'user'],
        annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
        all_caps_tag="wrap",
        fix_text=False,
        segmenter="english",
        corrector="english",
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[{"specialmasktoken": "[MASK]"}]).pre_process_doc
    return preprocessor



class DataClass(Dataset):
    def __init__(
            self, 
            args, 
            file, 
            file_path=False, 
            data_type="infer",
            label_type = "infer",
            pred_mode=False,
            preprocessor=None,
            pbar=tqdm):
        self.args = args

        self.pred_mode = pred_mode
        if self.pred_mode == False: 
            self.data, self.labels = self.load_dataset(file, file_path)
        else:
            self.data = file

        if not self.pred_mode:
            self.label_type = label_type
            if self.label_type == "infer":
                self.label_type = "binary" if ((self.labels == 1) | (self.labels == 0)).all() else "continuous"
            if self.label_type == "continuous":
                self.labels = 1 / (1 + np.exp(-self.labels))
            assert self.label_type == "continuous" or self.label_type == "binary"

        self.data_type = data_type
        if self.data_type == "infer":
            self.data_type = "utterance" if type(self.data[0]) == str else "conversation"
        assert self.data_type == "utterance" or self.data_type == "conversation"

        self.preprocessor = preprocessor if preprocessor else twitter_preprocessor()

        self.max_length = int(args['max_length'])

        self.bert_tokeniser = AutoTokenizer.from_pretrained(args["backbone"], do_lower_case=True)

        self.inputs, self.lengths, self.label_indices = self.process_data(pbar=pbar)

    def load_dataset(self, file, file_path):
        """
        :return: dataset after being preprocessed and tokenised
        """
        df = pd.read_csv(file) if file_path else file
        x_train, y_train = df.text.values, df.loc[:, "0":"10"].values
        return x_train, y_train

    def process_data(self, pbar=tqdm):
        """
        GO EMOTIONS
        segment_a = "admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervous optimism pride realization relief remorse sadness surprise or neutral?"
        label_names = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervous", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
        """

        segment_a = "anger anticipation disgust fear joy love optimism hopeless sadness surprise or trust?"
        label_names = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "hopeless", "sadness", "surprise", "trust"]

        if self.data_type == "conversation":
            segment_a = "Will speaker feel " + segment_a
            role_names = ("speaker", "respondent")

        inputs = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        lengths, label_indices = [], []
        for x in pbar(self.data):
            if self.data_type == "utterance":
                x = ' '.join(self.preprocessor(x))
            else:
                x = " [SEP] ".join(
                        f"{role_names[i % 2]}: " + " ".join(self.preprocessor(m)) + f"" for i, m in enumerate(x)
                    )

            x = self.bert_tokeniser.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                padding='max_length',
                                                truncation=True)
    
            input_id = x['input_ids']     
            input_mask = x['attention_mask']       
            input_length = len([i for i in input_mask if i == 1])
            inputs['input_ids'].append(input_id)
            inputs['attention_mask'].append(input_mask)
            inputs['token_type_ids'].append(x['token_type_ids'])
            lengths.append(input_length)

            #label indices
            label_idxs = [self.bert_tokeniser.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)

        inputs = {k: torch.tensor(inputs[k], dtype=torch.long) for k in inputs}
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)

        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = {k: self.inputs[k][index] for k in self.inputs}
        
        label_idxs = self.label_indices[index]
        length = self.lengths[index]

        if self.pred_mode == False:
            labels = self.labels[index]
            return inputs, labels, length, label_idxs
        else:
            return inputs, length, label_idxs

    def __len__(self):
        return len(self.lengths)