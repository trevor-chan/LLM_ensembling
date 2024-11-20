from meta_ai_api import MetaAI
import random
import time
from multiprocessing import Pool
import numpy as np
import json
import os
import replicate

token = open('token.txt').read().strip()
os.environ['REPLICATE_API_TOKEN'] = token

def query_ai_MetaAPI(expression):
    ai = MetaAI()
    message = f"Evaluate the following arithmetic expression and output the final answer as a single number. Do not show your work. Do not include any extra text. \n\n {expression} = ?"
    response = ai.prompt(message=message)
    return int(response['message'])

def query_ai_replicate(expression):
    # The meta/meta-llama-3-70b-instruct model can stream output as it's running.
    response = []
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 0,
            "top_p": 0.9,
            # "prompt": f"Evaluate the expression and output the answer as a single number. Do not show your work. Do not include any extra text. '%' is the modulo operator. \n\n {expression} = ?",
            "prompt": f"Evaluate the expression and output the answer as a single number. Do not show your work. Do not include any extra text. \n\n {expression} = ?",
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "system_prompt": "You are a helpful assistant",
            "length_penalty": 1,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "log_performance_metrics": False
        },
    ):
        response.append(str(event))
    return int(''.join(response).replace(',', ''))

def ask_ensemble(n,expression):
    responses = []
    count = 0
    while len(responses) < n:
        try:
            responses.append(query_ai_replicate(expression))
        except:
            time.sleep(1)
            pass
    return responses

def generate_ooo_expression(n_ops, max_int=9):
    """
    Recursively generates a mathematical expression with the specified number of operations.

    Parameters:
    n_ops (int): The number of operations (+, -, *) to include in the expression.
    n_ops (list): The number of operations (+, -, *) to include in the expression (randomly chosen from the list).

    Returns:
    str: A string representing the generated mathematical expression.
    """
    
    if type(n_ops) == list:
        n_ops = random.choice(n_ops)
    
    if n_ops == 0:
        # Base case: return a random positive integer between 1 and 9
        return str(random.randint(1, max_int))
    else:
        # Randomly split the remaining operations between left and right sub-expressions
        n_ops_left = random.randint(0, n_ops - 1)
        n_ops_right = n_ops - 1 - n_ops_left

        # Recursively generate left and right sub-expressions
        left_expr = generate_ooo_expression(n_ops_left, max_int)
        right_expr = generate_ooo_expression(n_ops_right, max_int)

        # Randomly choose an operator
        # op = random.choice(['+', '-', '*', '%'])
        op = random.choice(['+', '-', '*'])

        # Combine the expressions with the operator
        expr = f'{left_expr}{op}{right_expr}'

        # Randomly decide whether to wrap the expression in parentheses
        if random.choice([True, False]):
            expr = f'({expr})'
        return expr
    
def generate_multiplication_expression(out_digits=7):
    if type(out_digits) == list:
        out_digits = random.choice(out_digits)
    
    while True:
        middle_out_digits = random.choice(list(range(1, out_digits//2+1)))
        a = random.randint(10**(middle_out_digits - 1), 10**middle_out_digits - 1)
        b = random.randint(10**((out_digits - middle_out_digits) - 1), 10**(out_digits - middle_out_digits) - 1)
        if len(str(a*b)) == out_digits:
            break
    if random.choice([True, False]):
        expr = f'{b}*{a}'
    else:
        expr = f'{a}*{b}'
    return expr
    
def generate_and_evaluate(n_agents, n_ops, max_int, savedir=None):
    # expression = generate_ooo_expression(n_ops, max_int)
    expression = generate_multiplication_expression(max_int)
    responses = ask_ensemble(n_agents,expression)
    print(f'finished expression {expression}')
    if savedir is not None:
        with open(f'{savedir}/{expression}.json', "w") as outfile:
            json.dump({'expression':expression,
                       'responses':responses,
                       'n_agents':n_agents,
                       'n_ops':n_ops,
                       'max_int':max_int,
                       'correct':eval(expression),}, 
                      outfile, indent=4)
    return expression, responses
    
    
def take_exam(n_questions, n_agents, n_ops, max_int, savedir=None):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with Pool(8) as p:
        results = p.starmap(generate_and_evaluate, [(n_agents, n_ops, max_int, savedir) for _ in range(n_questions)])
    return results

def calc_mode(responses):
    mode = []
    max_count = responses.count(responses[0])
    for response in set(responses):
        if responses.count(response) > max_count:
            mode = [response]
            max_count = responses.count(response)
        elif responses.count(response) == max_count:
            mode.append(response)
    return mode, max_count

def vote(responses, threshold, correct):
    mode, count = calc_mode(responses)
    if len(mode) > 1:
        return 0
    elif count/len(responses) < threshold:
        return 0
    elif mode[0] == correct:
        return 1
    else:
        return -1

def vote_count(responses):
    return [int(np.sum(np.array(responses) == i)) for i in [-1, 0, 1]]

def calc_entropy(responses):
    entropy = []
    for response in responses:
        counts = np.array([response.count(i) for i in set(response)])
        probs = counts/np.sum(counts)
        entropy.append(-np.sum(probs*np.log(probs)))
    return entropy

def normalized_entropy(responses):
    entropy = []
    for response in responses:
        counts = np.array([response.count(i) for i in set(response)])
        probs = counts/np.sum(counts)
        h = -np.sum(probs*np.log(probs))
        entropy.append(h / np.log(len(response)))
    return entropy

def simple_entropy(responses):
    responses = np.array(responses).T
    return [len(np.unique(responses[:,i])) / len(responses) for i in range(responses.shape[1])]


if __name__ == '__main__':
    n_questions = 1000
    n_agents = 25
    n_ops = [2,3,4]
    max_int = 9
    results = take_exam(n_questions, n_agents, n_ops, max_int, savedir=f'math_outputs/test_exam_{n_questions}_{n_agents}_{n_ops}_{max_int}')