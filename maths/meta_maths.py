from meta_ai_api import MetaAI
import matplotlib.pyplot as plt
import tqdm
import random
import time
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pandas as pd

def query_ai(expression):
    ai = MetaAI()
    message = f"Evaluate the following arithmetic expression and output the final answer as a single number. Do not show your work. Do not include any extra text. \n\n {expression} = ?"
    response = ai.prompt(message=message)
    return int(response['message'])

def ask_ensemble(n,expression):
    responses = []
    # progress_bar = tqdm.tqdm(total=n)
    while len(responses) < n:
        try:
            responses.append(query_ai(expression))
            # progress_bar.update(1)
        except:
            pass
    return responses

def generate_expression(n_ops, max_int=9):
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
        left_expr = generate_expression(n_ops_left)
        right_expr = generate_expression(n_ops_right)

        # Randomly choose an operator
        op = random.choice(['+', '-', '*', '%'])

        # Combine the expressions with the operator
        expr = f'{left_expr}{op}{right_expr}'

        # Randomly decide whether to wrap the expression in parentheses
        if random.choice([True, False]):
            expr = f'({expr})'

        return expr
    
def generate_and_evaluate(n_agents, n_ops, max_int, savedir=None):
    expression = generate_expression(n_ops, max_int)
    responses = ask_ensemble(n_agents,expression)
    print(f'finished expression {expression}')
    if savedir is not None:
        json.save(f'{savedir}/{expression}.json', {'expression':expression,'responses':responses})    
    return expression, responses
    
    
def take_exam(n_questions, n_agents, n_ops, max_int, savedir=None):
    with Pool(16) as p:
        results = p.starmap(generate_and_evaluate, [(n_agents, n_ops, max_int, savedir) for _ in range(n_questions)])
    return results


if __name__ == '__main__':
    n_questions = 1000
    n_agents = 25
    n_ops = [2,3,4]
    max_int = 9
    results = take_exam(n_questions, n_agents, n_ops, max_int)
    pd.DataFrame(results, columns=['question','responses']).to_csv(f'{n_questions}questions_{n_agents}agents_{n_ops}operations_{max_int}max.csv')