import argparse
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hyperparameter_transfer(csv_file):
    widths = set()
    data = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            width = int(row['Width'])
            log2_lr = float(row['log2_LR'])
            loss = float(row['Loss'])
            widths.add(width)
            if width not in data:
                data[width] = {'log2_lr': [], 'loss': []}
            data[width]['log2_lr'].append(log2_lr)
            data[width]['loss'].append(loss)

    plt.figure(figsize=(12, 8))
    for width in widths:
        plt.plot(data[width]['log2_lr'], data[width]['loss'], marker='o', label=f'Width {width}')
    
    plt.xlabel('log2 Learning Rate', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.title('Maximal Update Parameterization', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('mup_hyperparameter_transfer.png')
    plt.close()

def plot_coordinate_check(csv_file):
    data = {}
    max_t = 0
    max_layer = 0
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            width = int(row['Width'])
            t = int(row['Iteration'])
            layer = int(row['Layer'])
            l1_norm = float(row['L1_Norm'])
            max_t = max(max_t, t)
            max_layer = max(max_layer, layer)
            if t not in data:
                data[t] = {}
            if layer not in data[t]:
                data[t][layer] = {'width': [], 'l1_norm': []}
            data[t][layer]['width'].append(width)
            data[t][layer]['l1_norm'].append(l1_norm)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('μP Transformer Coordinate Check', fontsize=20)

    for t in range(4):
        ax = axs[t // 2, t % 2]
        for layer in range(max_layer + 1):
            if layer in data[t]:
                ax.plot(data[t][layer]['width'], data[t][layer]['l1_norm'], marker='o', label=f'Layer {layer+1}')
        ax.set_xlabel('Width', fontsize=14)
        ax.set_ylabel('L1 Norm', fontsize=14)
        ax.set_title(f't={t+1}', fontsize=16)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.legend(fontsize=12)
        ax.grid(True)

        # Set x-axis ticks
        ax.set_xticks([2**i for i in range(6, 11)])
        ax.set_xticklabels([f'2^{i}' for i in range(6, 11)], fontsize=10)

        # Set y-axis ticks
        y_ticks = [2**i for i in [x / 10.0 for x in range(-4, 0, 1)]]  # Adjust this range as needed
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'2^{i}' for i in [x / 10.0 for x in range(-4, 0, 1)]], fontsize=10)

        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig('mup_coordinate_check.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot MuP results')
    parser.add_argument('plot_type', choices=['hyperparameter', 'coordinate'],
                        help='Type of plot to generate')
    parser.add_argument('csv_file', help='Path to the CSV file with data')
    args = parser.parse_args()

    if args.plot_type == 'hyperparameter':
        plot_hyperparameter_transfer(args.csv_file)
    elif args.plot_type == 'coordinate':
        plot_coordinate_check(args.csv_file)

    print(f"Plot for {args.plot_type} has been generated.")
