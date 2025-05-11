from utils.DataPruner import DataPruner
from utils.AssetsSelector import AssetSelector
from utils.WeigthOptimizer import WeigthOptimizer
import json
dp = DataPruner("./eth_hackathon/input_one_day.json")
dp.select_assets(15)
data = dp.data

assSel = AssetSelector(data, data['evaluation_date'])

# with open("prova.json", "w") as file:
#     json.dump(data, file)
assSel.create_qubo_model(10)
print(assSel.solve_qubo())

assets = assSel.get_selected_assets()

optim = WeigthOptimizer(assets)

print(optim.optimize_portfolio())