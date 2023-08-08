import datetime
import time
import os
import platform
from model_evaluation_utils import training_evaluation
from model_utils import create_phase1_model
from plot_keras_history import show_history, plot_history

phase1_model = create_phase1_model()
start_time = time.time()
history, phase1_model, results = training_evaluation(phase1_model, 10)
end_time = time.time()
runtime = end_time - start_time
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
print(results)
print(history.history)
plot_history(history, path=f"logs/{formatted_datetime}/history_plot.png")

# serialize model to JSON
model_json = phase1_model.to_json()
with open(f"logs/{formatted_datetime}/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
phase1_model.save_weights(f"logs/{formatted_datetime}/model.h5")
print("Saved model to disk")
