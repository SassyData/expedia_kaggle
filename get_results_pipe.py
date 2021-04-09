import pandas as pd
import sys
from proj_fns import create_submission

"""create submission useing sys.argv variables (model variables / submission name / model file)
e.g. if model variables = "models/top_12_feats.csv" + submission name = "rf_12vars.csv" + model = "rf_model.sav" 
then from terminal..  
$ python get_results_pipe.py "models/top_12_feats.csv" "new_submission_name.csv" "rf_model.sav"
 
 (model vars could also be None)
 """

# Set-up args
model_vars_file = sys.argv[1]
sub_file_name = sys.argv[2]
model_file = sys.argv[3]

# Raw data load
test_raw = pd.read_csv('data/test.csv')

# Load top 12 vars df
if model_vars_file:
    model_vars = pd.read_csv(model_vars_file)['0'].tolist()
    print(f"Model variables loaded: \n{model_vars}")
else:
    model_vars = None

# RF submission
create_submission(test_raw, model_vars, sub_file_name, model_file)


if __name__ == '__main__':
    # Load test set
    test_raw = pd.read_csv('data/test.csv')

    # Set-up args
    model_vars_file = "models/top_12_feats.csv"
    sub_file_name = "test_main_fn.csv"
    model_file = "rf_model.sav"

    # Load top 12 vars df
    model_vars = pd.read_csv(model_vars_file)['0'].tolist()
    print(f"Model variables loaded: \n{model_vars}")

    # RF submission
    create_submission(test_raw, model_vars, sub_file_name, model_file)


