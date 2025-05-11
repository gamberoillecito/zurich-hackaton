import portfolio

def run(input_data,solver_params,extra_arguments):
    if 'evaluation_date' in extra_arguments:
        input_data['evaluation_date']=extra_arguments['evaluation_date']
    solver = 'qubo'
    if 'ising' in extra_arguments:
        solver = 'ising'
    return portfolio.run(input_data, solver=solver)

# if __name__ == "__main__":
#     import json
#     with open("./eth_hackathon/input.json") as file:
#         data = json.load(file)
#     run(data , solver_params=None, extra_arguments={'evaluation_date': '2024-05-12'})