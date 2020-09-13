import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def get_results_from_path(path=None):
    if path is None:
        file_list = glob.glob('experiments/**/*.data.csv', recursive=True)
        latest_result = max(file_list, key=os.path.getctime)
        path = os.path.join(latest_result)
    print(path)
    if path.split('.')[-1] == 'pkl':
        df = pd.read_pickle(path)
    elif PATH.split('.')[-1] == 'csv':
        df = pd.read_csv(path)
    
    # df.loc[:,'round'] = df['round'] * (-1)
    df = df.sort_values('round', ascending=True)
    df = df.reset_index(drop=True)
    
    return df

def get_results_from_server(server):
    cs = server.metrics.metrics['cs']
    df = pd.DataFrame(cs)
    
    # df.loc[:,'round'] = df['round'] * (-1)
    df = df.sort_values('round', ascending=True)
    df = df.reset_index(drop=True)
    
    return df

def get_global_metric(df, mean='test_acc'):
    glb = df[df['c_name'] == 'global']
    data = glb[mean].to_numpy()
    return data

def get_mean_groupby(df, mean='train_loss', groupby='round'):
    lst = df.groupby(groupby)[mean].mean().to_numpy()
    return lst

def plot_by_round(df, plot_train=True, plot_test=True, plot_global=True):
    global_df = df[df['c_name'] == 'global']
    df = df[df['c_name'] != 'global']
    
    train_losses = get_mean_groupby(df, mean='train_loss', groupby='round')
    train_acc = get_mean_groupby(df, mean='train_acc', groupby='round')
    global_train_acc = get_global_metric(global_df, mean='train_acc')
    
    test_losses = get_mean_groupby(df, mean='test_loss', groupby='round')
    test_acc = get_mean_groupby(df, mean='test_acc', groupby='round')
    global_test_acc = get_global_metric(global_df, mean='test_acc')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=100, figsize=(6, 12))
    
    if plot_train:
        ax1.plot(train_losses, label = 'Train Loss (min={:.3f})'.format(np.min(train_losses)), color='C0')
        ax2.plot(train_acc, label = 'Train Accuracy (max={:.3f}, last={:.3f}) w.r.t. devices'.format(np.max(train_acc), train_acc[-1]), color='C0')
        if plot_global:
            ax3.plot(global_train_acc,
                     label='Train Accuracy (max={:.3f}, last={:.3f}) w.r.t data point'.format(np.max(global_train_acc), global_train_acc[-1]), color='C0')
    
    if plot_test:
        ax1.plot(test_losses, label = 'Test Loss (min={:.3f})'.format(np.min(test_losses)), color='C1')
        ax2.plot(test_acc, label = 'Test Accuracy (max={:.3f}, last={:.3f}) w.r.t. devices'.format(np.max(test_acc), test_acc[-1]), color='C1')
        if plot_global:
            ax3.plot(global_train_acc, 
                     label='Test Accuracy (max={:.3f}, last={:.3f}) w.r.t data point'.format(np.max(global_test_acc), global_test_acc[-1]), color='C1')

    
    plt.subplots_adjust(hspace=0.5)
    for ax in (ax1, ax2, ax3):
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=1)
    return fig

def fairness_result(df, subset='test'):
    acc = get_global_metric(df, mean=f'{subset}_acc')
    avg_acc = np.mean(acc)
    
    n_clients = len(df['c_name'].unique()) - 1
    max_r = max(df['round'])
    last_round = df[(df['round'] == max_r) & (df['c_name'] != 'global')]
    worst_10 = np.mean(last_round[f'{subset}_acc'].sort_values()[0:(int(n_clients*0.1))])
    best_10 = np.mean(last_round[f'{subset}_acc'].sort_values()[-(int(n_clients*0.1)):])
    var = np.var(last_round[f'{subset}_acc'])
    
    return {
        'subset': subset,
        'avg_acc': avg_acc * 100,
        'worst_10': worst_10 * 100,
        'best_10': best_10 * 100,
        'variance': var * 10000
    }

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g