#%%
## Import the config objects
from LSV1M_training.check_LSV1M_model.classical_exps_config import experiments_config, analyses_config, execute_function

## Perform the experiments
for exp in experiments_config :
    name_function = exp[0]
    params = exp[1]
    result = execute_function(name_function, params)

## Perform the analyses
for res in analyses_config :
    name_function = res[0]
    params = res[1]
    result = execute_function(name_function, params)
# %%

