# ANN4RAG

**Replication Package for SANER2025 Paper (Under Review)**

This repository provides the replication package for our paper on assertion generation, focusing on the use case of natural language processing (NLP) for generating test prompts.

**Dataset Links:**

* ATLAS: [https://sites.google.com/view/atlas-nmt/home](https://sites.google.com/view/atlas-nmt/home)
* Synthesis: [https://conala-corpus.github.io/](https://conala-corpus.github.io/)
* Commit:: [https://conala-corpus.github.io/](https://github.com/NNgen/nngen)


**Project Overview**

This project demonstrates the application of ANN4RAG for assertion generation using Codellama. It includes scripts for:

- Preprocessing data
- Constructing prompts from templates
- Running inference with Codellama
- Evaluating results using the ATLAS evaluation script (included)

**Installation**

1. **Codellama:** Follow the installation instructions for Codellama on your machine: [https://github.com/meta-llama/codellama](https://github.com/meta-llama/codellama)

**Running the Pipeline**

1. **Preprocessing (`construction/preprocess.py`):**
   - This script processes the provided datasets (ATLAS samples processed into JSONL format).
   - The resulting JSONL files will have a `code_candidates_tokens` field containing retrieved examples.

2. **Prompt Construction (`construction/construction.py`):**
   - This script populates pre-defined templates with preprocessed data to generate prompts.
   - The generated prompts will be stored in the `prompts` folder.

3. **Inference (`inference/codellama.py`):**
   - Use the following command to run inference with Codellama using two GPUs:

   ```bash
   torchrun --nproc_per_node 2 --master_port 29502 inference/codellama.py
