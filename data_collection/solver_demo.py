from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import glob
import argparse
import jsonlines
from datetime import datetime

from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from mcts_math.agents import SBSREACT
from mcts_math.agents import MCTS
from mcts_math.solver import Solver
from mcts_math.config import BaseConfig
from react_demo import load_qaf
from react_batch_demo import batch
import subprocess
import re
import time


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args.add_argument(
        "--qaf", "--question-answer-file", 
        type=str, 
        required=True,
        help="the file includes question / partial solution (optional) / answer (optional)")

    args = args.parse_args()
    return args

def find_matching_files(folder_path, qaf, mode, llm_version):
    prefix = f"{qaf}.{mode}.{llm_version}."
    
    pattern = os.path.join(folder_path, prefix + '*')
    matching_files = glob.glob(pattern)
    
    return matching_files


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    print(config)

    llm_version = os.path.basename(config.model_dir.rstrip("/"))

    data = load_qaf(args.qaf)
    if config.mode=='mcts':
        existing_files = find_matching_files('/'.join(args.qaf.split('/')[:-1]), args.qaf.split('/')[-1], config.mode, llm_version)
        number = [i.split(llm_version+'.')[-1].split('.jsonl')[0]  for i in existing_files]
        number = [int(i.split('_')[0])*int(i.split('_')[1])  for i in number]
        number.append(0)
        batch_count = int(max(number)/config.batch_size)
        print(batch_count)
        data = data[config.batch_size*batch_count:]


    solver = Solver(config=config)

    # init method
    if config.mode == "mcts":
        method = MCTS
    elif config.mode == "sbs":
        method = SBSREACT
    else:
        raise NotImplementedError
    
    for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
        if config.mode=='mcts':
            agents = [method(config=config, question=d["question"], test_cases=d["test"], ground_truth=d["answer"] if config.is_sampling else None)  #, success_rate=d["success_rate"]
                  for d in cur_data]
        else:
            if 'LiveCodeBench' in args.qaf:
                agents = [method(config=config, question=d["prompt"], test_cases=[], ground_truth=d["answer"] if config.is_sampling else None)  #, success_rate=d["success_rate"]
                  for d in cur_data]
            else:
                agents = [method(config=config, question=d["question"], test_cases=[], ground_truth=d["answer"] if config.is_sampling else None)  #, success_rate=d["success_rate"]
                  for d in cur_data]

        results = solver.solve(agents, config.mode=='mcts')
        
        if config.mode=='mcts':
            batch_count+=1
            saved_jsonl_file = f"{args.qaf}.{config.mode}.{llm_version}.{config.batch_size}_{batch_count}.jsonl" 
            with open(saved_jsonl_file, "w") as writer:
                for d in cur_data:
                    question = d["question"]
                    d["react"] = results[question]
                    writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                    writer.flush()


    if 'mbpp' in args.qaf or 'humaneval' in args.qaf or 'LiveCodeBench' in args.qaf:
        prediction = []
        for d in cur_data:
            if 'LiveCodeBench' in args.qaf:
                key= 'prompt'
            else:
                key = 'question'
            react = results[d[key]]
            if len(react['solutions']) == 0:
                #print(react)
                final_answer_states = []
                for state in react:
                    if isinstance(react[state], dict) and react[state]['final_answer']:
                        final_answer_states.append({'final_answer':react[state]['final_answer'], 'value':react[state]['value']})
                final_answer_states = sorted(final_answer_states, key=lambda x: x['value'], reverse=True)
                prediction.append({"task_id":d['task_id'], 'question':d[key], 'completion':final_answer_states[0]['final_answer']})
                #prediction.append({"task_id":d['task_id'], 'question':d[key], 'completion':''})
            else:
                completion = react['solutions'][0]["final_answer"].replace('```python','').replace('```','')
                #completion = react['solutions'][0]["final_answer"].split('```python')[1].split('```')[0]
                new_completion = []
                for line in completion.split('\n'):
                    if line.strip().startswith('assert'):
                        continue
                    new_completion.append(line)
                if 'humaneval' in args.qaf:
                    prediction.append({"task_id":d['task_id'], 'question':d[key], 'completion':d[key].split('```python')[1].split('def ')[0]+'\n'+'\n'.join(new_completion), 'original_completion':react['solutions'][0]})
                else:
                    prediction.append({"task_id":d['task_id'], 'question':d[key], 'completion':'\n'.join(new_completion), 'original_completion':react['solutions'][0]})
        saved_jsonl_file = f"{args.qaf}.{config.mode}.{llm_version}.prediction.jsonl"         
        with jsonlines.open(saved_jsonl_file,'w') as f:
            f.write_all(prediction)

