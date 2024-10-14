import time
from llama import Llama

generator = Llama.build(
        ckpt_dir='/home/pengfei/code/codellama-main/CodeLlama-13b-Instruct', 
        tokenizer_path='/home/pengfei/code/codellama-main/CodeLlama-13b-Instruct/tokenizer.model',
        max_seq_len=10000,
        max_batch_size=1)

class CodeLlama:

    def __init__(self,task):
        self.generator = generator
        self.task = task
    def get_suggestions(self,input_prompt):
        start_time = time.perf_counter()
        if self.task == "atlas":
            sys_content="Generate only assertions based on DEMO, and nothing else."
        elif self.task == "Conala":
            sys_content="Generate only single line code for the target requirement directly, and nothing else."
        elif self.task == "commit":
            sys_content="generate the response following the instruction"
        instructions = [
        [
            {
                "role": "system",
                "content": sys_content
            },
            {
                "role": "user",
                "content": str(input_prompt)
            }]
       ]
        results = self.generator.chat_completion(
        instructions, 
        max_gen_len=200,
        temperature=0,
        top_p=0.85,
    )
        response=results[0]['generation']['content']
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return round(run_time, 4), response
