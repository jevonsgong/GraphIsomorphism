import re
import matplotlib.pyplot as plt

# ==== MANUAL CONFIGURATION ====

# Dataset to list of (label, log_file_path) tuples
datasets_logs = {
    'CFI': [
        ('TraceNet', 'runs/log_Trace_cfi_False_3.txt'),
        ('GIN', 'runs/log_GIN_cfi_False_3.txt'),
    ],
    '3XOR': [
        ('TraceNet', 'runs/log_Trace_3xor_False_1.txt'),
        ('GIN', 'runs/log_GIN_3xor_False_8.txt'),
    ],
    'EXP': [
        ('TraceNet', 'runs/log_Trace_cfi_False_2.txt'),
        ('GIN', 'runs/log_GIN_exp_False_8.txt'),
    ],
    'SYN': [
        ('TraceNet', 'runs/log_Trace_syn_False_1.txt'),
        ('GIN', 'runs/log_GIN_syn_False_5.txt'),
    ],
    'SRG': [
        ('TraceNet', 'runs/log_Trace_sr_False_3.txt'),
        ('GIN', 'runs/log_GIN_sr_False_8.txt'),
    ],
}

# ==== REGEX FOR ACCURACY & F1 ====

acc_pattern = re.compile(r'val loss.*?acc (\d\.\d+)', re.IGNORECASE)
f1_pattern = re.compile(r'val loss.*?f1 (\d\.\d+)', re.IGNORECASE)

def extract_curve(log_path):
    accs, f1s = [], []
    with open(log_path, 'r') as f:
        for line in f:
            acc_match = acc_pattern.search(line)
            f1_match = f1_pattern.search(line)
            if acc_match: accs.append(float(acc_match.group(1)))
            if f1_match: f1s.append(float(f1_match.group(1)))
    return accs, f1s

# ==== PLOT PER DATASET ====

for dataset, entries in datasets_logs.items():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    acc_ax, f1_ax = ax

    for label, log_file in entries:
        acc_curve, f1_curve = extract_curve(log_file)
        acc_ax.plot(acc_curve, label=label)
        f1_ax.plot(f1_curve, label=label)

    acc_ax.set_title(f'{dataset} - Validation Accuracy')
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.legend()
    acc_ax.grid(True)

    f1_ax.set_title(f'{dataset} - Validation F1')
    f1_ax.set_xlabel('Epoch')
    f1_ax.set_ylabel('F1 Score')
    f1_ax.legend()
    f1_ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{dataset.lower()}_val_curves.png', dpi=300)
    plt.savefig(f'{dataset.lower()}_val_curves.pdf')
    plt.close()

    print(f'Saved plots for {dataset}')
