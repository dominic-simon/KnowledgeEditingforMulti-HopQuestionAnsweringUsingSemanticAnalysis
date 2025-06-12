from refined.inference.processor import Refined
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import warnings
import argparse
from tqdm import tqdm
from wikidata.client import Client

from CHECK import CHECK

def parse_args():
    '''
    Command line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', # LLM to be used
                        default='gpt-j',
                        choices=['gpt2-xl', 'gpt-j', 'vicuna-7b', 'vicuna-13b', 'falcon-7b', 'falcon-11b', 'mixtral'])
    parser.add_argument('--ds_name', # Dataset to be used
                        default='mquake-cf-3k',
                        choices=['mquake-cf-full', 'mquake-cf-3k', 'mquake-3k-v2', 'mquake-2002', 'mquake-t', 'mquake-hard'])
    parser.add_argument('--ds_start', # Dataset starting index
                        type=int,
                        default=0)
    parser.add_argument('--ds_end', # Dataset ending index
                        type=int,
                        default=3000)
    parser.add_argument('--num_new_tokens', # Number of tokens allowed to be generated per forward pass
                        type=int,
                        default=50)
    parser.add_argument('--similarity', # Embedding similarity to be used (dot product or cosine similarity)
                        default='cos',
                        choices=['cos', 'dot'])
    parser.add_argument('--sim_thresh', # Similarity threshold used when determining when an edit is sufficiently similar to the current subject, relationship
                        type=float,
                        default=0.8)
    parser.add_argument('--use_gpu', # True if you want to use a GPU, false if you want to run on a CPU
                        action='store_true')
    parser.add_argument('--gpu_num', # Choose the GPU number
                        type=int,
                        default=0)
    parser.add_argument('--verbose', # Print out accuracy as the dataset is evaluated
                        action='store_true')
    args = parser.parse_args()
    return args

def set_args(args):
    '''
    Parsing the command line arguments.
    '''
    args.device = f'cuda:{args.gpu_num}' if args.use_gpu else 'cpu'

    if args.model_name == 'gpt-j':
        args.model_name = 'EleutherAI/gpt-j-6B'
    elif args.model_name == 'vicuna-7b':
        args.model_name = 'lmsys/vicuna-7b-v1.5'
    elif args.model_name == 'vicuna-13b':
        args.model_name = 'lmsys/vicuna-13b-v1.5'
    elif args.model_name == 'falcon-7b':
        args.model_name = 'tiiuae/falcon-7b'
    elif args.model_name == 'falcon-11b':
        args.model_name = 'tiiuae/falcon-11b'
    elif args.model_name == 'mixtral':
        args.model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    
    if args.ds_name == 'mquake-cf-full':
        args.ds_name = 'MQuAKE-CF'
    elif args.ds_name == 'mquake-cf-3k':
        args.ds_name = 'MQuAKE-CF-3k'
    elif args.ds_name == 'mquake-2002':
        args.ds_name = 'MQuAKE-2002'
    elif args.ds_name == 'mquake-t':
        args.ds_name = 'MQuAKE-T'
    elif args.ds_name == 'mquake-hard':
        args.ds_name = 'MQuAKE-HARD'
    elif args.ds_name == 'mquake-3k-v2':
        args.ds_name = 'MQuAKE-CF-3k-V2'
    
    return args

def main(args):
    '''
    Example of how to set up CHECK in code.
    '''
    warnings.filterwarnings('ignore')

    # Initialize the LLM and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=False).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Initialize the entity linking model.
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers', entity_set="wikipedia")

    # Open all necessary files: the relationship type file and all prompts
    f = open('./mquake_relationships.txt', 'r')
    type_template = f.readlines()
    f.close()
    f = open('./prompts/type_extraction_prompt.txt', 'r')
    type_extraction_prompt = f.read()
    f.close()
    f = open('./prompts/sro_extraction_prompt.txt', 'r')
    extraction_prompt = f.read()
    f.close()
    f = open('./prompts/subquestion_prompt.txt', 'r')
    subq_prompt = f.read()
    f.close()
    f = open('./prompts/qa_prompt.txt', 'r')
    qa_prompt = f.read()
    f.close()

    # Initialize CHECK
    check = CHECK(model, 
                  tokenizer, 
                  refined,
                  type_template=type_template,
                  type_prompt=type_extraction_prompt,
                  extraction_prompt=extraction_prompt, 
                  subq_prompt=subq_prompt, 
                  qa_prompt=qa_prompt, 
                  is_vicuna='vicuna' in args.model_name,
                  similarity=args.similarity,
                  sim_thresh=args.sim_thresh,
                  device=args.device)

    # Load the dataset
    ds = json.load(open(f'./datasets/{args.ds_name}.json', 'r'))[args.ds_start : args.ds_end]

    # Extract all requested edits from the dataset
    client = Client()
    requested_edits = []
    edits_sentences = []
    for case in tqdm(ds[:], desc='Making Edit SROs'):
        for edit in case['requested_rewrite']:
            s = edit['subject']
            r = str(client.get(edit['relation_id'], load=True).label)
            o = edit['target_new']['str']       
            requested_edits.append((s, r, o))
            edits_sentences.append(edit['question'])
    
    # Pass the edits to check for embedding
    check.add_edits(requested_edits, edits_sentences)

    # Multi-hop Question Answering
    total = 0
    q_acc = 0
    i_acc = 0
    for d in tqdm(ds, desc=f'CHECK with {args.model_name} on {args.ds_name}'):
        case_correct = False
        total += 1

        # Try to answer each of the questions
        for q in d['questions']:
            try:
                final_answer = check.answer_question(q, num_new_tokens=args.num_new_tokens)
            except:
                if args.verbose:
                    print(f'Error at {total}')
                continue

            if args.verbose:
                print(final_answer, d['new_answer'], d['new_answer_alias'], '\n-----------------------------------------------\n')
            if final_answer == d['new_answer'] or final_answer in d['new_answer_alias']:
                q_acc += 1
                if not case_correct:
                    case_correct = True
                    i_acc += 1

        if args.verbose:
            print(f'\nDataset Size     : {len(ds)} Cases and {len(ds)*3} Questions')
            print(f'Case Accuracy: {i_acc} / {total} or {i_acc/total*100:.2f}')
            print(f'Question Accuracy: {q_acc} / {total*3} or {q_acc/(total*3)*100:.2f}\n')

if __name__ == '__main__':
    args = parse_args()
    args = set_args(args)
    main(args)