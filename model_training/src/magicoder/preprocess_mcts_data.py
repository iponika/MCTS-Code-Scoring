from dataclasses import dataclass, field
from typing import Literal, cast
import jsonlines
import os
import json
import random
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset,Dataset, concatenate_datasets
from transformers import HfArgumentParser
from collections import defaultdict

from prompt_template import SRC_INSTRUCT_INSTRUCTION_PROMPT
from utils import N_CORES, read_jsonl, write_jsonl

DatasetKey = Literal["evol-instruct", "codealpaca", "src-instruct", "combine"]

IGNORED_INDEX=-100
import ast

def check_syntax(code_string):
    try:
        code_string = code_string.split('```python')[1].split('```')[0]
        ast.parse(code_string)
        return True
    except:
        return False
    

@dataclass(frozen=True)
class Args:
    dataset_path: str
    key: DatasetKey
    output_file: str
    raw_dataset_path: str
    stage: str
    split: str = field(default="train")


def map_src_instruct(example: dict) -> dict:
    instructions = [
        SRC_INSTRUCT_INSTRUCTION_PROMPT.format(problem=problem)
        for problem in example["problem"]
    ]
    keys = [key for key in example.keys() if key not in ["problem", "solution"]]
    kwargs = {key: example[key] for key in keys}
    return dict(instruction=instructions, response=example["solution"], **kwargs)

def map_evol_instruct(example: dict) -> dict:
    instruction = example["instruction"]
    response = example["output"]
    return dict(
        instruction=instruction,
        response=response,
    )


def form_codealpaca_instruction(instruction: str, input: str) -> str:
    if input.strip() == "":
        return instruction
    return f"{instruction}\nInput: {input}"


def map_codealpaca(example: dict) -> dict:
    instruction = [
        form_codealpaca_instruction(instruction, input)
        for instruction, input in zip(example["instruction"], example["input"])
    ]
    response = example["output"]
    return dict(
        instruction=instruction,
        response=response,
    )


def map_mcts_instruct(example: dict) -> dict:
    instructions = [
        SRC_INSTRUCT_INSTRUCTION_PROMPT.format(problem=problem)
        for problem in example["problem"]
    ]
    keys = [key for key in example.keys() if key not in ["problem", "instruction", "solution", "q_value"]]
    kwargs = {key: example[key] for key in keys}
    
    return dict(instruction=instructions, response=example["solution"], q_value=example["q_value"], **kwargs)


def map_fn(example: dict, key: DatasetKey) -> dict:
    if key == "evol-instruct":
        return map_evol_instruct(example)
    elif key == "codealpaca":
        return map_codealpaca(example)
    elif key == "src-instruct":
        return map_mcts_instruct(example)
    elif key == "mcts":
        return map_mcts_instruct(example)
    else:
        raise ValueError(f"Unknown key: {key}")

def convert_punctuation(text):
    return text

def add_step_tags(text):
    result = text.replace('</step> \n<step>', '</step>\n<step>')
    for i in range(10):
        import re
        pattern = f"(?<!<step>)\n{i}"
        result = re.sub(pattern, f"\n</step>\n<step>\n{i}", result)
        pattern2 = f'(?<!<\/step>)\n<step>\n{i}'
        result =  re.sub(pattern2, f'\n</step>\n<step>\n{i}', result)
    return result


def validate_step_format(text):
    steps = text.strip().split('</step>')
    steps = [s.strip() for s in steps if s.strip()]
    
    if not steps:
        return False
        
    prev_num = None
    
    for i, step in enumerate(steps):
        if not step.startswith('<step>'):
            return False
            
        content = step[6:].strip()
        
        if '```' in content:
            if '<code>' not in content:
                return False
            if i!=len(steps)-1:
                return False
        else:
            first_line = content.split('\n')[0].strip()
            if not first_line or not first_line[0].isdigit():
                return False
                
            current_num = int(first_line[0])
            
            if prev_num is not None and current_num != prev_num + 1:
                return False
                
            prev_num = current_num

                
    return True



def mcts_data(files_path):
    files = []
    for root, dirs, fes in os.walk(files_path):
        for file in fes:
            if file.startswith('data_seed_test.json') and file.endswith('.jsonl') and 'all_fail' not in file and 'all_pass' not in file and len(file)>25:
                files.append(os.path.join(root, file))
    all_pass = 0
    all_fail = 0
    zero_valid = 0
    sample_number = 0
    filtered_data = 0
    
    all_fail_data = []
    all_pass_data = []
    pass_refine_data = []
    processed_data = []

    for path in tqdm(files):
        with jsonlines.open(path) as f:
            for example in f:
                reasoning_chain = example["react"]
            
                final_states = set()
                all_states = set(reasoning_chain.keys())
                
                for state in all_states:
                    if reasoning_chain[state]["q_value"]==0:
                        continue
                    is_final = True
                    for other_state in all_states:
                        if other_state.startswith(state + '.'):
                            is_final = False
                            break
                    if is_final:
                        final_states.add(state)
                count1=0
                count2=0
                current_data = []
                existing_data = []
                for state in final_states:
                    response = []
                    q_value = []
                    bad=False
                    name = ''
                    for choice in state.split('.'):
                        name = name+'.'+choice if len(name)>0 else choice
                        reasoning_chain[name]["text"] = reasoning_chain[name]["text"].replace('</code>\n<step>',  '</code>\n</step>')
                        if len(reasoning_chain[name]["text"])!=0: 
                            if (name!=state and (not reasoning_chain[name]["text"].strip().endswith('</step>\n<step>') or '</code>' in reasoning_chain[name]["text"] or '<code>' in reasoning_chain[name]["text"])) or \
                               (name!=state and reasoning_chain[state]["q_value"]==1 and (not reasoning_chain[name]["text"].strip()[0].isdigit())) or \
                               (name==state and not reasoning_chain[name]["text"].strip().endswith('</code>\n</step>'))  or \
                               (name==state and not reasoning_chain[name]["text"].strip().startswith('<code>')):
                                bad=True
                                filtered_data+=1
                                break
                            reasoning_chain[name]["text"] = add_step_tags(convert_punctuation(reasoning_chain[name]["text"]))
                            response.append(reasoning_chain[name]["text"])
                            q_value.append(reasoning_chain[name]["q_value"])
                            
                    if bad or response[-1] in existing_data:
                        continue
                
                    if q_value[-1]==1:
                        if count1>3:
                            continue
                        else:
                            count1+=1
                    if q_value[-1]==-1:
                        if count2>3:
                            continue
                        else:
                            count2+=1
                    existing_data.append(response[-1])
                    current_data.append(dict(problem=convert_punctuation(example["question"]), solution=response, q_value=q_value))

                if count1==0:
                    if count2==0:
                        all_fail_data+=[{'problem':example["question"],'incorrect_path':'','response':[],'q_value':[],'answer':example["answer"]}]*6
                        zero_valid+=1
                    else:
                        all_fail_data+=[{'problem':example["question"],'incorrect_path':'<step>\n'+'\n'.join(i['solution']),'response':i['solution'],'q_value':i['q_value'],'answer':example["answer"]} for i in current_data]
                        all_fail+=1
                elif count2==0:
                    all_pass+=1
                    all_pass_data+=current_data
                    i=current_data[0]
                    number = random.randint(0,len(i['solution'])-1)
                    pass_refine_data.append({'question':example["question"].split('.')[0]+'[pass_expand]'+'\n'.join(i['solution'][:number]),'original_question':example["question"],'test':example['test'],'answer':example["answer"],'q_value':i['q_value'],'number':number})
                
                processed_data+=current_data
                sample_number+=1


    print(zero_valid,all_pass,all_fail,sample_number)

    with jsonlines.open(os.path.join(files_path, 'all_fail.jsonl'),'w') as f:
        f.write_all(all_fail_data)
    with open(os.path.join(files_path, 'all_pass.json'), 'w', encoding='utf-8') as f:
        json.dump(pass_refine_data, f, indent=4)
    
    
    pos_samples = [item for item in processed_data if item["q_value"][-1] == 1]
    neg_samples = [item for item in processed_data if item["q_value"][-1] == -1]
    print(len(pos_samples),len(neg_samples),filtered_data)

    sample_size = min(len(pos_samples), len(neg_samples))
    sampled_pos = random.sample(pos_samples, sample_size)
    sampled_neg = random.sample(neg_samples, sample_size)
    
    balanced_data = sampled_pos + sampled_neg
    random.shuffle(balanced_data)


    with open(os.path.join(files_path, 'all.json'), 'w', encoding='utf-8') as f:
        for item in balanced_data:
            if 'random.' in item['solution'][-1] and item['q_value'][-1]==1:
                continue
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')



def fix_step_numbers(text):
    import re
    step_pattern = r'(<step>\s*)(\d+)(\.)'
    first_step = re.search(step_pattern, text)
    if first_step and first_step.group(2) == '0':
        def increment_number(match):
            return f'{match.group(1)}{int(match.group(2)) + 1}{match.group(3)}'
        
        text = re.sub(step_pattern, increment_number, text)
    
    return text

def combine(files_path):
    processed_data = []
    with jsonlines.open(os.path.join(files_path, 'data_mcts.jsonl')) as f:
        for i in f:
            i['q_value']=list(map(float, i['q_value']))
            processed_data.append({'problem':i['instruction'], 'solution':i['response'], 'q_value':i['q_value']})


    files = []
    for root, dirs, fes in os.walk(files_path):
        for file in fes:
            if file.startswith('all_pass.')  and file.endswith('.jsonl'):
                files.append(os.path.join(root, file))
    expand_data = []
    for path in tqdm(files):
        with jsonlines.open(path) as f:
            for example in f:
                reasoning_chain = example["react"]
            
                final_states = set()
                all_states = set(reasoning_chain.keys())
                
                for state in all_states:
                    if reasoning_chain[state]["q_value"]==0:
                        continue
                    is_final = True
                    for other_state in all_states:
                        if other_state.startswith(state + '.'):
                            is_final = False
                            break
                    if is_final:
                        final_states.add(state)
                        
                count=0
                existing_data = []
                for state in final_states:
                    response = []
                    q_value = []
                    bad=False
                    name = ''
                    for choice in state.split('.'):
                        name = name+'.'+choice if len(name)>0 else choice
                        reasoning_chain[name]["text"] = reasoning_chain[name]["text"].replace('</code>\n<step>',  '</code>\n</step>')
                        if len(reasoning_chain[name]["text"])!=0: 
                            if (name!=state and (not reasoning_chain[name]["text"].strip().endswith('</step>\n<step>') or '</code>' in reasoning_chain[name]["text"] or '<code>' in reasoning_chain[name]["text"])) or \
                               (name==state and not reasoning_chain[name]["text"].strip().endswith('</code>\n</step>'))  or \
                               (name==state and not reasoning_chain[name]["text"].strip().startswith('<code>')):
                                bad=True
                                break
                            reasoning_chain[name]["text"] = add_step_tags(convert_punctuation(reasoning_chain[name]["text"]))
                            response.append(reasoning_chain[name]["text"])
                            q_value.append(reasoning_chain[name]["q_value"])

                    if ''.join(response).count('<step>')>=len(q_value) or bad or response[-1] in existing_data:
                        continue
                
                    if q_value[-1]==-1:
                        if count>5:
                            continue
                        else:
                            count+=1
                        existing_data.append(response[-1])
                        response = [i+'<step>' for i in example["question"].split('[pass_expand]')[1].split('<step>')[:example["number"]]]+response
                        expand_data.append(dict(problem=convert_punctuation(example["original_question"]), solution=response, q_value=example["q_value"][:int(example["number"])]+q_value))


    refine_data = []
    refined_data = []
    with jsonlines.open(os.path.join(files_path, 'refined_fail.jsonl')) as f:
        for i in f:
            refine_data.append(i)
    count=0
    existing_res=[]
    for i in tqdm(refine_data):
        if i['res'] in existing_res:
            continue
        else:
            existing_res.append(i['res'])
        if len(i['response'])>0:
            if not validate_step_format('<step>\n'+'<step>'.join(i['res'].split('<step>')[1:])) or 'Now its your turn:' in i['res']:
                continue
            refined_path = '<step>'.join(i['res'].split('<step>')[1:])
            if refined_path.strip().startswith('<code>'):
                response = i['response']
                code = refined_path.split('<code>')[1].split('</code>')[0].strip()
                if check_syntax(code):
                    response[-1] = response[-1].split('<code>')[0].strip()+'\n<code>\n'+code+'\n</code>\n</step>'
                else:
                    response[-1] = response[-1].split('<code>')[0].strip()+'\n<code>\n```python\n'+i["answer"].strip()+'\n```\n</code>\n</step>'
                q_value = [-100]*(len(i['q_value'])-1)+[1]
            else:
                refined_path = fix_step_numbers('<step>\n'+refined_path).strip('<step>\n')
                try:
                    step_number = refined_path.strip().split('.')[0]
                    assert f'<step>\n{step_number}' in i["incorrect_path"]
                except:
                    continue
                combine_path = i["incorrect_path"].split(f'<step>\n{step_number}')[0]+'<step>'+refined_path
                combine_path = combine_path.strip('<step>')
                response = []
                q_value = []
                for r,q in zip(i["response"],i["q_value"]):
                    if r not in combine_path:
                        break
                    combine_path = combine_path.split(r)[1].strip()
                    response.append(r)
                    q_value.append(-100)
                code = combine_path.split('<code>')[1].strip()
                if check_syntax(code):
                    combine_path = combine_path.split('<code>')[0]+'\n<code>\n'+code+'\n</code>\n</step>'
                else:
                    combine_path = combine_path.split('<code>')[0]+'\n<code>\n```python\n'+i["answer"].strip()+'\n```\n</code>\n</step>'
                response.append(combine_path.strip())
                q_value.append(1)
            refined_data.append({'problem':i["problem"], 'solution':response, 'q_value': q_value})
            for item in processed_data:
                if '<step>\n'+'\n'.join(item['solution']) == i['incorrect_path']:
                    for idx in range(len(q_value)):
                        if q_value[idx]==-100:
                            item['q_value'][idx]=-100
        else:
            if not validate_step_format('<step>\n'+'<step>'.join(i['res'].split('<step>')[1:])) or 'Now its your turn:' in i['res']:
                continue
            refined_path = '<step>'.join(i['res'].split('<step>')[1:]).split('<code>')[0].strip()
            if not refined_path.startswith('1'):
                continue
            code = i['res'].split('<code>')[1].strip()
            if check_syntax(code):
                refined_path += '\n<code>\n'+code+'\n</code>\n</step>'
            else:
                refined_path += '\n<code>\n```python\n'+i["answer"].strip()+'\n```\n</code>\n</step>'
            refined_data.append({'problem':i["problem"], 'solution':[refined_path], 'q_value': [1]})
    processed_data.extend(refined_data)


    pos_samples = [item for item in processed_data if item["q_value"][-1] == 1 or item["q_value"][-1] == -100]
    neg_samples = [item for item in processed_data if item["q_value"][-1] == -1]
    print(len(pos_samples),len(neg_samples))    
    sample_size = min(len(pos_samples), len(neg_samples))
    sampled_pos = random.sample(pos_samples, k=sample_size)
    sampled_neg = random.sample(neg_samples, k=sample_size)
    balanced_data = sampled_pos + sampled_neg
    random.shuffle(balanced_data)


    with open(os.path.join(files_path, 'all.json'), 'w', encoding='utf-8') as f:
        for item in balanced_data:
            if 'random.' in item['solution'][-1] and item['q_value'][-1]==1:
                continue
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def stage2(files_path):
    old_data = []
    with jsonlines.open(os.path.join(files_path, 'data_pr.jsonl')) as f:
        for i in f:
            i['q_value']=list(map(float, i['q_value']))
            old_data.append({'problem':i['instruction'], 'solution':i['response'], 'q_value':i['q_value']})

    processed_data = []
    with jsonlines.open(os.path.join(files_path, 'direct_generation.jsonl')) as f:
        for i in f:
            for p,r in zip(i["predictions"], i["res"]):
                if not r:
                    processed_data.append({'problem':i["question"], 'solution':['<code>\n```python\n'+p.strip()+'\n```\n</code>\n</step>\n'], 'q_value': [-1]})
                else:
                    processed_data.append({'problem':i["question"], 'solution':['<code>\n```python\n'+p.strip()+'\n```\n</code>\n</step>\n'], 'q_value': [1]})
    pos_samples = [item for item in processed_data if item["q_value"][-1] == 1]
    neg_samples = [item for item in processed_data if item["q_value"][-1] == -1]
    print(len(pos_samples),len(neg_samples))    
    sample_size = min(len(pos_samples), len(neg_samples))
    sampled_pos = random.sample(pos_samples, k=sample_size)
    sampled_neg = random.sample(neg_samples, k=sample_size)
    balanced_data = sampled_pos + sampled_neg


    sample_size = min(len(balanced_data), len(old_data))
    sampled_new = random.sample(balanced_data, k=sample_size)
    sampled_old = random.sample(old_data, k=sample_size)
    balanced_data = sampled_new + sampled_old

    with open(os.path.join(files_path, 'all.json'), 'w', encoding='utf-8') as f:
        for item in balanced_data:
            if 'random.' in item['solution'][-1] and item['q_value'][-1]==1:
                continue
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    if args.key == "combine":
        assert args.dataset_path == "json" and args.data_files is not None
        all_data: list[dict] = []
        for data_file in args.data_files:
            data = read_jsonl(data_file)
            all_data.extend(
                dict(instruction=item["instruction"], response=item["response"])
                for item in data
            )
        write_jsonl(args.output_file, all_data)
    else:
        if args.stage=='mcts':
            mcts_data(args.raw_dataset_path)
        elif args.stage=='pr':
            combine(args.raw_dataset_path)
        elif args.stage=='s2':
            stage2(args.raw_dataset_path)

        dataset = load_dataset(
            "json",
            data_files=os.path.join(args.raw_dataset_path, 'all.json'),
            split=args.split,
            num_proc=N_CORES,
        )
        dataset = dataset.map(
            map_fn,
            fn_kwargs=dict(key=args.key),
            batched=True,
            num_proc=N_CORES,
            remove_columns=dataset.column_names,
        )
        with Path(args.output_file).open("w", encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
    data = []
