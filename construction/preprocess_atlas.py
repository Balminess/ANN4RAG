import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jsonlines
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from annoy import AnnoyIndex
from rank_bm25 import BM25Okapi,BM25L
import faiss
import time
import numpy as np

def write_to_files(processed,output_path,file_name):
    with jsonlines.open(os.path.join(output_path, file_name),'w') as f:
        f.write_all(processed)



def rankbm25_preprocess(train, test, output_path, number):
    tokenized_corpus = [doc.split(" ") for doc in question]
    bm25 = BM25Okapi(tokenized_corpus)
    
    processed = []
    start_time = time.perf_counter()
    for obj in tqdm(test, total=len(test)):
        query = (obj['focal_method']+obj['test_method']).split(" ")
        score = bm25.get_scores(query)
        rtn =np.argsort(score)[::-1][:number] 
        code_candidates_tokens = []
        for i in rtn:
            code_candidates_tokens.append({'focal_method': focal_method[i],'test_method': test_method[i],'assertion': answer[i],'idx':int(i),
                                           'test_name': test_name[i],'method_name':method_name[i]})        
        processed.append({'focal_method': obj['focal_method'],'test_method': obj['test_method'],'assertion': obj['assertion'],'method_name':obj['method_name'], 'test_name': obj['test_name'],'code_candidates_tokens': code_candidates_tokens})
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"bmExecution time: {execution_time:.6f} seconds")

    write_to_files(processed,output_path,'rankbm25_'+str(number)+'.jsonl')

def bm25l_preprocess(train, test, output_path, number):
    tokenized_corpus = [doc.split(" ") for doc in question]
    bm25 = BM25L(tokenized_corpus)
    
    processed = []
    start_time = time.perf_counter()
    for obj in tqdm(test, total=len(test)):
        query = (obj['focal_method']+obj['test_method']).split(" ")
        score = bm25.get_scores(query)
        rtn =np.argsort(score)[::-1][:number] 
        code_candidates_tokens = []
        for i in rtn:
            code_candidates_tokens.append({'focal_method': focal_method[i],'test_method': test_method[i],'assertion': answer[i],'idx':int(i),
                                           'test_name': test_name[i],'method_name':method_name[i]})        
        processed.append({'focal_method': obj['focal_method'],'test_method': obj['test_method'],'assertion': obj['assertion'],'method_name':obj['method_name'], 'test_name': obj['test_name'],'code_candidates_tokens': code_candidates_tokens})
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"bmExecution time: {execution_time:.6f} seconds")
    write_to_files(processed,output_path,'bm25L_'+str(number)+'.jsonl')


def annoy_preprocess(train, test, output_path, number):
    processed = []
    dim = 768  

    annoyindex = AnnoyIndex(dim, 'dot')

    for idx in range(len(train)):
        annoyindex.add_item(idx, question_emb[idx])
    # annoyindex.build(tree_num)
    annoyindex.build(10)

    start_time = time.perf_counter()
    for obj in tqdm(test):  
        query = obj['focal_method']+obj['test_method']
        query_emb = model.encode(query)
        hits=annoyindex.get_nns_by_vector(query_emb,number)
        code_candidates_tokens = []
        for i in hits:
            code_candidates_tokens.append({'focal_method': focal_method[i],'test_method': test_method[i],'assertion': answer[i],'idx':int(i),
                                           'test_name': test_name[i],'method_name':method_name[i]})
        processed.append({'focal_method': obj['focal_method'],'test_method': obj['test_method'],'assertion': obj['assertion'],'method_name':obj['method_name'], 'test_name': obj['test_name'],'code_candidates_tokens': code_candidates_tokens})
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"AnnoyExecution time: {execution_time:.6f} seconds")
    write_to_files(processed,output_path,'atlas_annoy_'+str(number)+'.jsonl')

def lsh_preprocess(train, test, output_path, number):


    index=faiss.index_factory(768, "LSH",faiss.METRIC_L2)

    processed = []
    index.add(question_emb)
    start_time = time.perf_counter()
    for obj in tqdm(test):
        query = obj['focal_method']+obj['test_method']
        query_emb = model.encode(query)
        D, hits =index.search(query_emb.reshape(1,-1), number)
        code_candidates_tokens = []
        for i in hits[0]:
            code_candidates_tokens.append({'focal_method': focal_method[i],'test_method': test_method[i],'assertion': answer[i],'idx':int(i),
                                           'test_name': test_name[i],'method_name':method_name[i]})
        processed.append({'focal_method': obj['focal_method'],'test_method': obj['test_method'],'assertion': obj['assertion'],'method_name':obj['method_name'], 'test_name': obj['test_name'],'code_candidates_tokens': code_candidates_tokens})
        
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"lshExecution time: {execution_time:.6f} seconds")
    write_to_files(processed,output_path,'atlas_lsh_'+str(number)+'.jsonl')

def hnsw_preprocess(train, test, output_path, number):
    
    index =faiss.index_factory(768, "HNSW",faiss.METRIC_INNER_PRODUCT)
    index.add(question_emb)
    processed = []
    start_time = time.perf_counter()
    for obj in tqdm(test):
        query = obj['focal_method']+obj['test_method']
        query_emb = model.encode(query)
        D, hits =index.search(query_emb.reshape(1,-1), number)
        code_candidates_tokens = []
        for i in hits[0]:
            code_candidates_tokens.append({'focal_method': focal_method[i],'test_method': test_method[i],'assertion': answer[i],'idx':int(i),
                                           'test_name': test_name[i],'method_name':method_name[i]})
        processed.append({'focal_method': obj['focal_method'],'test_method': obj['test_method'],'assertion': obj['assertion'],'method_name':obj['method_name'], 'test_name': obj['test_name'],'code_candidates_tokens': code_candidates_tokens})
        
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"hnswExecution time: {execution_time:.6f} seconds")
    write_to_files(processed,output_path,'atlas_hnsw_'+str(number)+'.jsonl')


def sbert_preprocess(train, test, output_path, number):
    processed = []
    start_time = time.perf_counter()
    for obj in tqdm(test):
        query = obj['focal_method']+obj['test_method']
        query_emb = model.encode(query)

        hits = util.semantic_search(query_emb, question_emb, top_k=number)[0]
        code_candidates_tokens = []
        for i in range(len(hits)):

            code_candidates_tokens.append({'focal_method': focal_method[hits[i]['corpus_id']],'test_method': test_method[hits[i]['corpus_id']],'assertion': answer[hits[i]['corpus_id']],'idx':int(hits[i]['corpus_id']),
                                           'test_name': test_name[hits[i]['corpus_id']],'method_name':method_name[hits[i]['corpus_id']]})
        processed.append({'focal_method': obj['focal_method'],'test_method': obj['test_method'],'assertion': obj['assertion'],'method_name':obj['method_name'], 'test_name': obj['test_name'],'code_candidates_tokens': code_candidates_tokens})      
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"sbExecution time: {execution_time:.6f} seconds")
    write_to_files(processed,output_path,'sbert_'+str(number)+'.jsonl')



if __name__ == "__main__":
    train = []
    test=[]
   
    with jsonlines.open('/home/pengfei/code/ANN4RAG/dataset/atlas/atlas-train-demo.jsonl') as f:     
        for i in f:
            if i['focal_method']!="":
                train.append(i)
    
    with jsonlines.open('/home/pengfei/code/ANN4RAG/dataset/atlas/atlas-test-demo.jsonl') as f:
        for i in f:
            if i['focal_method']!="":
                test.append(i)
    question = [obj['focal_method']+obj['test_method'] for obj in train] 
    answer = [obj["assertion"] for obj in train]

    focal_method = [obj['focal_method'] for obj in train]
    test_method = [obj['test_method'] for obj in train]
    method_name = [obj['method_name'] for obj in train]
    test_name = [obj['test_name'] for obj in train]

    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    question_emb = model.encode(question)

    numbers=64

    sbert_preprocess(train, test,'/home/pengfei/code/ANN4RAG/preprocess', numbers) 
 
 


