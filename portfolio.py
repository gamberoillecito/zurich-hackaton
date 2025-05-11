from utils.DataPruner import DataPruner
from utils.AssetsSelector import AssetSelector
from utils.WeigthOptimizer import WeigthOptimizer
import json

def run(input_data, solver='qubo'):
    if solver == 'qubo':
        dp = DataPruner(input_data)
        data = dp.select_assets(200)
        # print(data['evaluation_date'])
        assSel = AssetSelector(data, data['evaluation_date'])

        assSel.compute_qubo_model()
        assets = assSel.solve_qubo()
    else:
        dp = DataPruner(input_data)
        data = dp.select_assets(14)
        # print(data['evaluation_date'])
        assSel = AssetSelector(data, data['evaluation_date'])

        assSel.compute_cost_hamiltonian()
        assSel.compute_ansatz()
        assets = assSel.solve_ising()

    optim = WeigthOptimizer(assets)

    with open("output.json", 'w') as f:
        json.dump(optim.optimize_portfolio(), f)

    # dp = DataPruner(input_data)
    # data = dp.select_assets(10)
    # # print(data['evaluation_date'])
    # assSel = AssetSelector(data, data['evaluation_date'])

    # # with open("prova.json", "w") as file:
    # #     json.dump(data, file)
    # assSel.compute_cost_hamiltonian()
    # assSel.compute_ansatz()
    # assets = assSel.solve_ising()

    # optim = WeigthOptimizer(assets)

    # print(optim.optimize_portfolio())