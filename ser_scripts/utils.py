import numpy as np
import os
from bidict import bidict
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay,
)


def compute_WA_UA(pred, labels, label2id_merged):
    pred = pred if isinstance(pred, list) else pred.tolist()
    labels = labels if isinstance(labels, list) else labels.tolist()
    label2id = bidict(label2id_merged)

    label_correct = { label: 0 for label in label2id.keys() }
    label_tot = { label: labels.count(num) for label, num in label2id.items() }

    for i in range(len(pred)):
        if pred[i] == labels[i]:
            label_correct[label2id.inverse[pred[i]]] += 1

    label_acc = { label: label_correct[label] / label_tot[label] for label in label2id.keys() }

    UA = np.mean(list(label_acc.values()))
    WA = np.sum(list(label_correct.values())) / len(labels)
    
    return WA, UA, label_acc


def plot_pie_chart(items, wandb_log=False, prefix=''):
    fig1, ax1 = plt.subplots()
    labels = sorted(items.keys())
    values = [items[key] for key in labels]
    _, texts, _ = ax1.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    #[text.set_color('white') for text in texts]
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    if wandb_log:
        f_name = prefix + 'piechart_label_dist'
        save_path = f_name + '.png'
        plt.savefig(save_path, facecolor='White', transparent=True)
        im = plt.imread(save_path)
        wandb.log({f_name : wandb.Image(im, caption='Label Distribution - ' + prefix[:-1].capitalize() + os.linesep + str(items))})
    
    plt.show()
    
def generate_reverse_dict(label2id):
    label2id_merged = label2id
    key_len = len(set(label2id.keys()))
    val_len = len(set(label2id.values()))
    if key_len != val_len:
        combo_labels = [[k for k,v in label2id.items() if v == y] for y in set(label2id.values())]
        label2id_merged = {
            '/'.join(labels): label2id[labels[0]] for labels in combo_labels            
        }
    id2label = {
        v: k for k, v in label2id_merged.items()
    }
    return label2id_merged, id2label


def count_df_by_column(df, col):
    return {col_val: df[col].tolist().count(col_val) for col_val in set(df[col])}

def print_metadata(df, wandb_log=False, prefix=''):
    label_count = count_df_by_column(df, 'emo')
    print('Number of samples: ' + str(len(df)))
    print(label_count)
    plot_pie_chart(label_count, wandb_log=wandb_log, prefix=prefix)

def plot_log_save_confusion_matrix(mat, axis, prefix, norm):
    postfix = '_norm' if norm else ''
    f_name = prefix + 'confusion_matrix' + postfix
    ConfusionMatrixDisplay(mat, display_labels=axis).plot(cmap='Blues', colorbar=True)
    plt.savefig(f_name + '.png', facecolor='White', transparent=True)
    plt.show()
    
    im = plt.imread(f_name + '.png')
    wandb.log({f_name : wandb.Image(im, caption='Confusion Matrix' + ' - Normalized' if norm else '')})
    
def report_result_confusion_matrix(output, axis, prefix):
    pred = np.argmax(output.predictions, axis=-1)
    mat = confusion_matrix(output.label_ids, pred)
    mat_norm = confusion_matrix(output.label_ids, pred, normalize='true')

    # Plot, save and log confusion matrixes for the entire test-set
    plot_log_save_confusion_matrix(mat=mat, axis=axis, prefix=prefix, norm=False)
    plot_log_save_confusion_matrix(mat=mat_norm, axis=axis, prefix=prefix, norm=True)