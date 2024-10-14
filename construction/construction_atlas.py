import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jsonlines
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, RobertaModel
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def instance_selection(method, number, insturction, input_insturction, output_insturction):
    preprocess_file="/home/pengfei/code/ANN4RAG/preprocess/sbert_64.jsonl"

    output_path="/home/pengfei/code/ANN4RAG/prompts/"
    test = []
    with jsonlines.open(preprocess_file) as f:
        for i in f:
            test.append(i)
    prompts = []
    
    for obj in test:
        topk = []
        for sample in obj['code_candidates_tokens'][:number]:
                topk.append(sample)
        prompt = ''

        topk.reverse()
        if number:
            for sample in topk:
                prompt += insturction
                prompt += input_insturction+'\n'
                prompt += sample['focal_method'].strip()+'\n'
                prompt +='### UNIT_TEST'+'\n'
                prompt += sample['test_method'].strip()+'\n'
                prompt +='[METHOD_UNDER_TEST]:'+ sample['method_name']+'\n'
                prompt +='[UNIT_TEST]:'+sample['test_name']+'\n'
                prompt += output_insturction+'\n'
                prompt += sample['assertion'].strip()+'\n'
                prompt +="end_of_demo"+'\n'
                prompt += '\n\n'
        else: prompt=""
        tmp_prompt = prompt + '### METHOD_UNDER_TEST'+'\n'
        tmp_prompt += obj['focal_method']+'\n'
        tmp_prompt +='### UNIT_TEST'+'\n'
        tmp_prompt +=obj['test_method']+'\n'
        tmp_prompt +='[METHOD_UNDER_TEST]:'+ obj['method_name']+'\n'
        tmp_prompt +='[UNIT_TEST]:'+obj['test_name']+'\n'
        tmp_prompt += output_insturction
        prompts.append({'prompt':tmp_prompt, 'label':obj['assertion']})

    with jsonlines.open(os.path.join(output_path, 'atlas_'+str(method)+'_prompts_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(prompts)


if __name__ == "__main__":

    insturction = ''
    input_insturction = '### METHOD_UNDER_TEST'
    output_insturction = '### generate assertion'
 
    instance_selection('sbert', 1, insturction, input_insturction, output_insturction) 
    

    
