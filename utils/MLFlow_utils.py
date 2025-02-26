import mlflow
import json

def save_configuration(out_file:str, data_to_save:dict):
    d = data_to_save

    json_object = json.dumps(d)
    # mlflow.log_params(data_to_save)

    with open(out_file, "w") as outfile:
        outfile.write(json_object)