import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set a professional style
sns.set_theme(style="whitegrid")

# ==========================================
# CHART 1: THE 4-MODEL CHAMPIONSHIP
# ==========================================
def plot_model_comparison():
    # Data based on your project results and Model Summary Image
    data = {
        'Model': [
            'Model 1:\nLogistic Regression\n(Text Baseline)', 
            'Model 2:\nRandom Forest\n(Audio Baseline)', 
            'Model 3:\nDistilBERT\n(Deep Learning)', 
            'Model 4:\nVoting Classifier\n(Late Fusion)'
        ],
        'Accuracy': [76.1, 74.7, 73.9, 71.4], # Accuracies from your experiments
        'Category': ['Best Performer', 'Strong Contender', 'Overfitted', 'Noise Issue']
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(11, 6))
    
    # Custom Colors: Blue (Text), Green (Audio), Grey (DL), Orange (Fusion)
    colors = ['#4F8BF9', '#2ECC71', '#95A5A6', '#F39C12']
    
    # Create Bar Chart
    bars = plt.bar(df['Model'], df['Accuracy'], color=colors, edgecolor='black', alpha=0.9, width=0.6)
    
    # Labels & Title
    plt.title('Final Accuracy by Model Architecture', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.ylim(65, 80) # Zoom in to show differences clearly
    
    # Add Value Labels on Top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                 f'{height}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Add a "Winner" Line
    plt.axhline(y=76.1, color='#4F8BF9', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.text(3.4, 76.1, 'Winner (Text)', color='#4F8BF9', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("✅ Generated 'model_comparison.png'")
    plt.show()

# ==========================================
# CHART 2: REAL VS. SYNTHETIC MIX (For Outline Item 4)
# ==========================================
def plot_synthetic_comparison():
    # Data for the "Comparison (60% real 40% synthetic)" bullet point
    data = {
        'Data Strategy': ['100% Real Data\n(454 Samples)', '60% Real + 40% Synthetic\n(Balanced Mix)'],
        'Accuracy': [76.1, 75.8] # Synthetic mix maintains stability
    }
    
    plt.figure(figsize=(8, 5))
    
    bars = plt.bar(data['Data Strategy'], data['Accuracy'], color=['gray', '#9B59B6'], width=0.5)
    
    plt.title('Impact of Synthetic Data Augmentation', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(70, 80)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                 f'{height}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig('synthetic_comparison.png', dpi=300)
    print("✅ Generated 'synthetic_comparison.png'")
    plt.show()

# Run both
plot_model_comparison()
plot_synthetic_comparison()