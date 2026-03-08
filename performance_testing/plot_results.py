import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_sparsity_chart():
    csv_path = "performance_testing/results/sparsity_benchmark.csv"
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['sparsity'] * 100, df['fps'], marker='o', linewidth=2, color='#00aaff')
    plt.fill_between(df['sparsity'] * 100, df['fps'], alpha=0.2, color='#00aaff')
    
    plt.title("Residue Core: Throughput vs Signal Sparsity", fontsize=14, fontweight='bold')
    plt.xlabel("Signal Sparsity (%)", fontsize=12)
    plt.ylabel("Frames Per Second (FPS)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')
    
    # Highlight 99% point
    v99 = df[df['sparsity'] >= 0.99].iloc[0]
    plt.annotate(f"{v99['fps']:,.0f} FPS", 
                 xy=(99, v99['fps']), xytext=(90, v99['fps']*1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    os.makedirs("performance_testing/results/charts", exist_ok=True)
    plt.savefig("performance_testing/results/charts/sparsity_throughput.png", dpi=300)
    print("Generated: sparsity_throughput.png")

def plot_scaling_chart():
    csv_path = "performance_testing/results/scaling_benchmark.csv"
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df['model_size'].astype(str), df['speedup'], color='#7700ff', alpha=0.7)
    
    plt.title("PyTorch Shield: Speedup vs Model Hidden Size", fontsize=14, fontweight='bold')
    plt.xlabel("Model Complexity (Linear Hidden Dim)", fontsize=12)
    plt.ylabel("Relative Speedup (x-factor)", fontsize=12)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    plt.legend()
    
    plt.savefig("performance_testing/results/charts/speedup_scaling.png", dpi=300)
    print("Generated: speedup_scaling.png")

if __name__ == "__main__":
    # Settings for "Paper Style" charts
    plt.style.use('bmh')
    plot_sparsity_chart()
    plot_scaling_chart()
