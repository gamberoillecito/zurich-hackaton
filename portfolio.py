from utils.DataPruner import DataPruner
from utils.AssetsSelector import AssetSelector
from utils.WeigthOptimizer import WeigthOptimizer

def run(input_data):
    dp = DataPruner(input_data)
    data = dp.select_assets(10)
    # print(data['evaluation_date'])
    assSel = AssetSelector(data, data['evaluation_date'])

    # with open("prova.json", "w") as file:
    #     json.dump(data, file)
    assSel.compute_cost_hamiltonian()
    assSel.compute_ansatz()
    assets = assSel.solve()

    optim = WeigthOptimizer(assets)

    print(optim.optimize_portfolio())