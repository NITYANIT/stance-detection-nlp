
import torch
import torch.nn as nn
import pandas as pd
import re
import wordninja
import preprocessor as p
from transformers import AutoTokenizer, AutoModel, AutoConfig

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
LABEL_MAP  = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}

def data_clean(strings, norm_dict={}):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.RESERVED)
    clean_data = p.clean(strings)
    clean_data = re.sub(r"#SemST", "", clean_data)
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    clean_data = [[x.lower()] for x in clean_data]
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict:
            clean_data[i] = norm_dict[clean_data[i][0]].split()
            continue
        if clean_data[i][0].startswith("#") or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])
    return [j for i in clean_data for j in i]

def prepare_input(topic, text, norm_dict={}, add_prefix=True):
    clean_text   = data_clean(text, norm_dict)
    topic_str    = ("I am in favor of " + topic + " ! ") if add_prefix else topic
    clean_target = data_clean(topic_str, norm_dict)
    return " ".join(clean_text), " ".join(clean_target)

class roberta_large_classifier(nn.Module):
    def __init__(self, num_labels=3, dropout=0.1):
        super().__init__()
        self.config  = AutoConfig.from_pretrained("vinai/bertweet-base")
        self.roberta = AutoModel.from_pretrained("vinai/bertweet-base")
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()
        self.linear  = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        self.out     = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        last_hidden   = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        eos_token_ind = input_ids.eq(self.config.eos_token_id).nonzero()
        assert len(eos_token_ind) == 3 * len(input_ids)
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i % 3 == 0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i + 1) % 3 == 0]
        x_atten_clone = attention_mask.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, attention_mask, x_atten_clone):
            att[begin:]  = 0;  att2[:begin+2] = 0
            att[0]       = 0;  att2[end]      = 0
        txt_l      = attention_mask.sum(1).to(attention_mask.device)
        topic_l    = x_atten_clone.sum(1).to(attention_mask.device)
        txt_vec    = attention_mask.type(torch.FloatTensor).to(attention_mask.device)
        topic_vec  = x_atten_clone.type(torch.FloatTensor).to(attention_mask.device)
        txt_mean   = torch.einsum("blh,bl->bh", last_hidden[0], txt_vec)   / txt_l.unsqueeze(1)
        topic_mean = torch.einsum("blh,bl->bh", last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        cat    = torch.cat((txt_mean, topic_mean), dim=1)
        query  = self.dropout(cat)
        linear = self.relu(self.linear(query))
        return self.out(linear)

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model     = roberta_large_classifier(num_labels=3, dropout=0.1)
    state     = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"Model ready on {DEVICE}")
    return tokenizer, model

def predict_single(tokenizer, model, topic, text):
    clean_text, clean_target = prepare_input(topic, text)
    encoding = tokenizer(clean_text, clean_target, add_special_tokens=True,
                         max_length=MAX_LENGTH, padding="max_length",
                         truncation=True, return_tensors="pt")
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs    = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    result   = {"topic": topic, "text": text, "prediction": LABEL_MAP[pred_idx]}
    for idx, label in LABEL_MAP.items():
        result[f"prob_{label}"] = round(probs[idx], 4)
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--topic",      required=True)
    parser.add_argument("--text",       required=True)
    args = parser.parse_args()
    tokenizer, model = load_model(args.model_path)
    result = predict_single(tokenizer, model, args.topic, args.text)
    print(f"Prediction : {result['prediction']}")
    for label in LABEL_MAP.values():
        print(f"P({label}) : {result[f'prob_{label}']}")
