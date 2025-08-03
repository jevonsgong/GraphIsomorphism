import re
import matplotlib.pyplot as plt

# ==== MANUAL CONFIGURATION ====

# Each entry: (Label for legend, path to log file)
tracenet_configs = [
    ('depth64_width2', 'runs/log_Trace_cfi_False_1_64_2.txt'),
    ('depth128_width1', 'runs/log_Trace_cfi_False_3.txt'),
    ('depth32_width1', 'runs/log_Trace_cfi_False_2.txt'),
]

# ==== REGEX EXTRACTION ====

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

# ==== PLOT ====

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
acc_ax, f1_ax = ax

for label, log_file in tracenet_configs:
    acc_curve, f1_curve = extract_curve(log_file)
    acc_ax.plot(acc_curve, label=label)
    f1_ax.plot(f1_curve, label=label)

acc_ax.set_title('TraceNet - Validation Accuracy (by config)')
acc_ax.set_xlabel('Epoch')
acc_ax.set_ylabel('Accuracy')
acc_ax.legend()
acc_ax.grid(True)

f1_ax.set_title('TraceNet - Validation F1 (by config)')
f1_ax.set_xlabel('Epoch')
f1_ax.set_ylabel('F1 Score')
f1_ax.legend()
f1_ax.grid(True)

plt.tight_layout()
plt.savefig('tracenet_config_comparison.png', dpi=300)
plt.close()

print("Saved config comparison plots.")
