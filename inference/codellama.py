import jsonlines
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from codex_api import CodeLlama

codex_api = CodeLlama('commit')

def codex(nn_method,number,task):

    inference_time=0

    filename='/home/pengfei/code/ANN4RAG/prompts/atlas_sbert_prompts_1.jsonl'
    result_filename='/home/pengfei/code/ANN4RAG/results/atlas_sbert_result_1.jsonl'
    
    querys = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            querys.append(obj)
    # querys=querys[:100]
    data=[]
    for pos in tqdm(range(len(querys))):
        query = querys[pos]

        (inference_time, response_actual) = codex_api.get_suggestions(query['prompt'])  
        inference_time+=inference_time
        result = {}
        result['label'] = query['label']
        result['actual'] = response_actual
        result['idx'] = pos
        data.append(result)
    with jsonlines.open(result_filename, mode='w') as f:
        f.write_all(data)
    
    return inference_time/len(querys)
 

if __name__ == "__main__":  
    task='atlas' 
    codex('sbert',1,task)