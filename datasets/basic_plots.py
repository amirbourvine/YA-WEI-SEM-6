import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import earthpy.plot as ep

import os

def print_basic_statistics(df, y):
    print("Shape is: ", df.shape)
    summary = pd.DataFrame([{'class_name': val, 'samples': np.sum(y == val)} for val in range(1,14)])
    print(summary)
    print(summary.sum())

def read_dataset(filename, gt=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir,filename))
    df.drop(df.columns[0], axis=1, inplace = True)
    
    return df

def plot_gt(df):
    arr = df.to_numpy()
    plt.matshow(arr, cmap=plt.cm.viridis)
    plt.axis('off')
    plt.colorbar()
    plt.show()

def plot_composite(df):
    bands_num = df.shape[-1]
    if bands_num==102:
        data = df.to_numpy().reshape(((1096, 715,102)))
    elif bands_num==103:
        data = df.to_numpy().reshape((((610,340, 103))))
    elif bands_num==176:
        data = df.to_numpy().reshape((((512, 614, 176))))

    data = np.moveaxis(data, 2, 0)
    ep.plot_rgb(data, rgb=(36, 17, 11), title='Composite Image of Pavia University', figsize=(10, 8))
    plt.show()

def plot_spectral_band(df, num=4):
    bands_num = df.shape[-1]
    arr_list = []
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens',  'Reds']
    for i in range(num):
        c = np.random.randint(bands_num)
        arr = df[f'{c}'].to_numpy()
        if bands_num==102:
            arr_list.append((arr.reshape((1096, 715)),c))
        elif bands_num==103:
            arr_list.append((arr.reshape((610, 340)),c))
        elif bands_num==176:
            arr_list.append((arr.reshape((512, 614)),c))
        elif bands_num==126:
            arr_list.append((arr.reshape((670, 605)),c))

    _, axs = plt.subplots(1,num, sharex=True, sharey=True, figsize=(15, 15))
    plt.tight_layout(pad=5.0)
    
    if num == 1:
        axs  = [axs]

    for i in range(num):
        arr,c = arr_list[i]
        im = axs[i].matshow(arr, cmap=cmaps[i%len(cmaps)])
        axs[i].axis('off')
        axs[i].title.set_text(f"Band - {c}")
        plt.colorbar(im,ax=axs[i],fraction=0.08, pad=0.04)
    plt.show()

