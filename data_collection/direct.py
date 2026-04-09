
from typing import Optional, Any, Dict, List, Callable, Type, Tuple
import os
import sys
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput
import jsonlines
import json
from subprocess import TimeoutExpired,Popen,PIPE
import psutil
import tempfile
import os



GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')

PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. 
@@ Instruction
Write a solution to the following problem, please only write solution code and do not write any test code:
{instruction}

@@ Response
{response}"""

def test_pass(test_case, func_str):
    try:
        func_name = func_str.split('def')[1].split('(')[0].strip()
        refined_test = []
        for line in test_case.splitlines():
            if line.startswith('from tmp import') or line.startswith('from your_module import'):
                continue
            if 'import' in line and func_name in line:
                continue
            refined_test.append(line)
        extracted_test = '\n'.join(refined_test)
    
        code = func_str + "\n" + extracted_test
        
        def kill(proc_pid:int):
            process= psutil.Process(proc_pid)
            
            for proc in process.children(recursive=True):
                proc.kill()
            process.kill()
        
        def pexec(strcmd:str,n_timeout:int):
            try:
                p=Popen(strcmd,shell=True, stdout=PIPE, stderr=PIPE)
                stdout, stderr = p.communicate(timeout=n_timeout)
                return stdout.decode('utf-8'), stderr.decode('utf-8'), p.returncode
            except TimeoutExpired:
                kill(p.pid)
                return None, None,-1
            except Exception as e:
                return None, None,-1
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir='/tmp') as f:
            f.write(code.encode('utf-8'))
            temp_path = f.name
    
        temp_dir = os.path.dirname(temp_path)
        cmd = f'cd {temp_dir} ; python {temp_path}'
        stdout, stderr, returncode = pexec(cmd, 3)
        
        os.unlink(temp_path)
        
        test_pass = returncode == 0 and 'AssertionError' not in stderr
        return test_pass
    except:
        return False



def equl(grt, prd):
    for g in grt:
        if not test_pass(g, prd):
            return False
    return True

def direct_generation(path):
    with open(os.path.join(path, 'data_seed_test.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompts = []
    for d in data:
        prompts.append(PROMPT.format(instruction=d["question"], response='```python'))

    llm = LLM(
        model='deepseek-ai/deepseek-coder-6.7b-instruct', 
        tensor_parallel_size=len(GPUS), 
        trust_remote_code=True,
        seed=42,
        swap_space=24,
        max_model_len=7000,
        gpu_memory_utilization=0.9
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        use_beam_search=False,
        max_tokens=512, 
        n=5,
        best_of=5,
        stop=['```','\n```','```\n','```\n\n','```\n\n\n']
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params) 
    
    for i,o in tqdm(zip(data,outputs)):
        i['predictions'] = []
        for n in range(5):
            i['predictions'].append(o.outputs[n].text)
        i['predictions'] = list(set(i['predictions']))

        pass_all = True
        i['res'] = []
        for pre in i['predictions']:
            i['res'].append(equl(i['test'], pre))
            if not i['res'][-1]:
                pass_all = False
        i['pass_all'] = pass_all
    with jsonlines.open(os.path.join(path, 'direct_generation.jsonl'), 'w') as f:
        f.write_all(data)



path = sys.argv[1]
direct_generation(path)

