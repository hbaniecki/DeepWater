#%%
import os
import numpy as np
import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torcheval.metrics as metrics


#%%
df          = pd.read_csv('data/processed/monthly.csv')
df_metadata = pd.read_csv('data/processed/monthly_metadata.csv')


#%%
target_feature   = "o"
threshold_ncount = 10
threshold_time   = "2009-01"

# train and valid are disjoint subsets of gauges
gauge_ids_train = df_metadata.gauge_id[(df_metadata.time_start < threshold_time) & 
                                       (df_metadata.ncount >= threshold_ncount)].values
gauge_ids_valid = df_metadata.gauge_id[(df_metadata.time_start >= threshold_time) & 
                                       (df_metadata.ncount >= threshold_ncount)].values

df_train = df[(df.gauge_id.isin(gauge_ids_train)) & (df.year_month < threshold_time)]
df_test  = df[(df.gauge_id.isin(gauge_ids_train)) & (df.year_month >= threshold_time)]
df_valid = df[(df.gauge_id.isin(gauge_ids_valid))]

def preprocess_data(X):
    return torch.log10(torch.nan_to_num(X)+1)


#%% 
# split a multivariate sequence into samples
def split_gauge_sequence(sequence, target, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(sequence.shape[0]):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_ix].drop(target, axis=1)
        seq_y = sequence[end_ix-1:out_end_ix][target]
        X.append(seq_x.values)
        y.append(seq_y.values)
    return torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float()

# split sequences for all gauges and combine into one dataset
def split_gauges_sequences(df, target, n_steps_in, n_steps_out):
    X_train_list, y_train_list = [], []
    for gauge_id in df.gauge_id.unique():
        X_train_gauge, y_train_gauge = split_gauge_sequence(
            df[df.gauge_id == gauge_id].drop(["gauge_id", "year_month"], axis=1), 
            target=target, 
            n_steps_in=n_steps_in, 
            n_steps_out=n_steps_out
        )
        X_train_list.append(X_train_gauge)
        y_train_list.append(y_train_gauge)
    return torch.vstack(X_train_list), torch.vstack(y_train_list)

n_steps_in = 12
n_steps_out = 1

X_train, y_train = split_gauges_sequences(
    df_train, target="o", 
    n_steps_in=n_steps_in, n_steps_out=n_steps_out
)
X_test, y_test = split_gauges_sequences(
    df_test, target="o", 
    n_steps_in=n_steps_in, n_steps_out=n_steps_out
)
X_valid, y_valid = split_gauges_sequences(
    df_valid, target="o", 
    n_steps_in=n_steps_in, n_steps_out=n_steps_out
)

loader_train = data.DataLoader(data.TensorDataset(
    preprocess_data(X_train), y_train
    ), shuffle=True, batch_size=1024)
loader_test = data.DataLoader(data.TensorDataset(
    preprocess_data(X_test), y_test
    ), shuffle=True, batch_size=1024)
loader_valid = data.DataLoader(data.TensorDataset(
    preprocess_data(X_valid), y_valid
    ), shuffle=True, batch_size=1024)


#%%
class LSTM(nn.Module):
    def __init__(
            self, 
            input_size,
            hidden_size=128, 
            num_layers=1, 
            dropout=0, 
            output_size=1
        ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.output_size = output_size
    def forward(self, x):
        x, _ = self.lstm(x)
        # extract only the last time step
        x = x[:, (x.shape[1]-self.output_size):x.shape[1], :]
        x = self.linear(x).flatten(1)
        return x
 
model = LSTM(
    input_size=20, 
    hidden_size=128, 
    num_layers=1, 
    dropout=0.3, 
    output_size=n_steps_out
)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
metric_train, metric_test, metric_valid = metrics.R2Score(), metrics.R2Score(), metrics.R2Score()


#%%
n_epochs = 300
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader_train:
        not_nan = ~y_batch.isnan()
        if torch.all(y_batch.isnan()):
            continue
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred[not_nan], y_batch[not_nan].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0 or epoch == n_epochs-1:
        model.eval()    
        with torch.no_grad():
            loss_train, loss_test, loss_valid = 0, 0, 0
            for X_batch, y_batch in loader_train:
                not_nan = ~y_batch.isnan()
                if torch.all(y_batch.isnan()):
                    continue
                y_pred = model(X_batch.to(device))
                loss_train += loss_fn(y_pred[not_nan], y_batch[not_nan].to(device)).item()
                metric_train.update(y_pred[not_nan].to('cpu'), y_batch[not_nan].to('cpu'))
            for X_batch, y_batch in loader_test:
                not_nan = ~y_batch.isnan()
                if torch.all(y_batch.isnan()):
                    continue
                y_pred = model(X_batch.to(device))
                loss_test += loss_fn(y_pred[not_nan], y_batch[not_nan].to(device)).item()
                metric_test.update(y_pred[not_nan].to('cpu'), y_batch[not_nan].to('cpu'))
            for X_batch, y_batch in loader_valid:
                not_nan = ~y_batch.isnan()
                if torch.all(y_batch.isnan()):
                    continue
                y_pred = model(X_batch.to(device))
                loss_valid += loss_fn(y_pred[not_nan], y_batch[not_nan].to(device)).item()
                metric_valid.update(y_pred[not_nan].to('cpu'), y_batch[not_nan].to('cpu'))
        
        print("Epoch %d | Train RMSE: %.2f, R2: %.3f | Out-of-time RMSE: %.2f, R2: %.3f | Out-of-distribution RMSE: %.2f, R2: %.3f" % 
              (epoch+1, 
               np.sqrt(loss_train / len(loader_train)), 
               metric_train.compute().item(),
               np.sqrt(loss_test / len(loader_test)), 
               metric_test.compute().item(),
               np.sqrt(loss_valid / len(loader_valid)), 
               metric_valid.compute().item()))
        

#%%
newpath = f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}'
if not os.path.exists(newpath):
    os.makedirs(newpath)

torch.save(model.state_dict(), f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/model.pt')
torch.save(loader_train, f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_train.pt')
torch.save(loader_test, f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_test.pt')

# %%
loader_train_mini = data.DataLoader(data.TensorDataset(
    preprocess_data(X_train), y_train
    ), shuffle=True, batch_size=2)
torch.save(loader_test, f'models/target_feature={target_feature}_threshold_ncount={threshold_ncount}_threshold_time={threshold_time}_n_steps_in={n_steps_in}_n_steps_out={n_steps_out}/loader_train_mini.pt')
