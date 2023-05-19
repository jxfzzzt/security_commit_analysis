import datetime
import torch

message_model_name = 'bert-base-uncased'
code_model_name = 'microsoft/codebert-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_path = "logs_train" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
save_path = "saved_dict/bert-baseline-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".pth"
metrics_path = "metrics_data-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"
dropout = 0.15
max_len = 38262
batch_size = 64
epoch = 100
sampling_rate = 16000
lr = 0.01
step_size = 15
gamma = 0.95
num_classes = 5
DATA_PATH = '../../data'
