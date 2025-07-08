'''
Some function reproduced and some functions adapted from the Github Repository for MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions.
Repository Link: https://github.com/princeton-nlp/MQuAKE
Function File Link: https://github.com/princeton-nlp/MQuAKE/blob/main/run_mello.ipynb
'''

import torch

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_sent_embeddings(sents, contriever, tok, device, BSZ=32):    
    all_embs = []
    for i in range(0, len(sents), BSZ):
        sent_batch = sents[i:i+BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs

def retrieve_similarities(query_emb, fact_embs, device='cpu'):
    #sim = (query_emb.to(device) @ fact_embs.T.to(device))
    sim = torch.nn.functional.cosine_similarity(query_emb.to(device), fact_embs.to(device))
    return sim

def retrieve_similarities_cos(query_emb, fact_embs, device='cpu'):
    sim = torch.nn.functional.cosine_similarity(query_emb.to(device), fact_embs.to(device))
    return sim

def retrieve_similarities_dot(query_emb, fact_embs, device='cpu'):
    sim = (query_emb.to(device) @ fact_embs.T.to(device))
    return sim
