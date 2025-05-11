from utils.DataPruner import DataPruner
from utils.AssetsSelector import AssetSelector
from utils.WeigthOptimizer import WeigthOptimizer
import json
import logging
logger = logging.getLogger(__name__)

with open("./eth_hackathon/input_one_day.json") as file:
    data = json.load(file)
dp = DataPruner(data)
data = dp.select_assets(14)
# print(data['evaluation_date'])
assSel = AssetSelector(data, data['evaluation_date'])

######################
## SOLVE WITH ISING ##
######################

assSel.compute_cost_hamiltonian()
assSel.compute_ansatz()
assets = assSel.solve_ising()

#####################
## SOLVE WITH QUBO ##
#####################
# assSel.compute_qubo_model()
# assets = assSel.solve_qubo()

optim = WeigthOptimizer(assets)

with open("output.json", 'w') as f:
    json.dump(optim.optimize_portfolio(), f)