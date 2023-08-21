import datetime
import time
import os
import platform
from utils.evaluation_utils import training_evaluation
from utils.model_utils import create_model
from plot_keras_history import plot_history


def save_eval_results(history, results, runtime, model):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")
    parent_folder_name = "logs"
    os.makedirs(parent_folder_name, exist_ok=True)
    nested_folder_name = os.path.join(parent_folder_name, formatted_datetime)
    os.makedirs(nested_folder_name, exist_ok=True)

    file_path = os.path.join(nested_folder_name, "output.txt")
    with open(file_path, "w") as file:
        file.write(f"{platform.uname()._asdict()}\n")
        file.write(f"Runtime: {runtime:.6f} seconds\n")
        file.write(f"{history.history}\n")
        file.write(f"{results}\n")
    plot_history(history, path=f"logs/{formatted_datetime}/history_plot.png")

    # serialize model to JSON
    model_json = model.to_json()
    with open(f"logs/{formatted_datetime}/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save(f"logs/{formatted_datetime}/model.h5")
    print("Saved model to disk")


def run_model_training_and_eval(epochs):
    model = create_model()
    start_time = time.time()
    history, model, results = training_evaluation(model, epochs)
    end_time = time.time()
    runtime = end_time - start_time
    save_eval_results(history, results, runtime, model)
    return history, model, results


run_model_training_and_eval(10)
