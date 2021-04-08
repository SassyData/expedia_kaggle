import pandas as pd
import sys
from proj_fns import preprocess_step, create_submission

"""create submission useing sys.argv variables (model variables / submission name / model file)
e.g. if model variables = "models/top_12_feats.csv" + submission name = "rf_12vars.csv" + model = "rf_model.sav" 
then from terminal..  
$ python get_results_pipe.py "models/top_12_feats.csv" "rf_12vars.csv" "rf_model.sav"
 
 (model vars could also be None)
 """

# Raw data load
test_raw = pd.read_csv('data/test.csv')

# Load top 12 vars df
if sys.argv[0]:
    model_vars = pd.read_csv(sys.argv[0])['0'].tolist()
    print(f"Model variables loaded: \n{model_vars}")

# RF submission
create_submission(test_raw, model_vars = sys.argv[0], sub_file_name = sys.argv[1], model_file = sys.argv[2])


if __name__ == '__main__':
    test_raw = pd.read_csv('data/test.csv')

    # Load top 12 vars df
    model_vars = pd.read_csv("models/top_12_feats.csv")['0'].tolist()
    print(f"Model variables loaded: \n{model_vars}")

    # RF submission
    create_submission(test_raw, model_vars, sub_file_name = "test_main_fn.csv", model_file = "rf_model.sav")


