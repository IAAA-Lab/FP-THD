#!/usr/bin/env python3
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_log(path):
    train_logs = []
    val_logs = []
    
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            
            # Training: "Iter : 100 	 LR : 0.00010 	 training loss : 205.33746" 
            if "Iter :" in line and "training loss :" in line:
                iter_match = re.search(r'Iter : (\d+)', line)
                loss_match = re.search(r'training loss : ([\d.]+)', line)
                if iter_match and loss_match:
                    train_logs.append({
                        'iter': int(iter_match.group(1)),
                        'loss': float(loss_match.group(1))
                    })
            
            # Validation: "Val. loss : 5.273 	 CER : 0.0229 	 WER : 0.0830"
            elif "Val. loss :" in line:
                val_match = re.search(r'Val\. loss : ([\d.]+)', line)
                cer_match = re.search(r'CER : ([\d.]+)', line)
                wer_match = re.search(r'WER : ([\d.]+)', line)
                if val_match and cer_match and wer_match:
                    val_logs.append({
                        'val_loss': float(val_match.group(1)),
                        'cer': float(cer_match.group(1)),
                        'wer': float(wer_match.group(1))
                    })

    print(f"Parsed {len(train_logs)} training points, {len(val_logs)} validation points")
    
    train_df = pd.DataFrame(train_logs)
    val_df = pd.DataFrame(val_logs)
    
    if train_df.empty:
        print("No training data found!")
        sys.exit(1)
        
    return train_df, val_df

def main():
    if len(sys.argv) != 2:
        print("Usage: python training_figure.py <logfile>")
        sys.exit(1)

    log_path = sys.argv[1]
    train_df, val_df = parse_log(log_path)
    
    # Convert to numpy arrays before plotting (avoids pandas/matplotlib bug)
    x_train = train_df["iter"].to_numpy()
    y_train = train_df["loss"].to_numpy()
    
    # Training figure (SEPARATE)
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(x_train, y_train, 'b-', linewidth=2, label="Training loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss (min: {y_train.min():.1f}, final: {y_train[-1]:.1f})")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("training_loss.png", dpi=300, bbox_inches='tight')
    fig1.savefig("training_loss.pdf", bbox_inches='tight')
    
    # Validation figure (SEPARATE)
    if not val_df.empty:
        val_steps = np.arange(len(val_df))

        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.plot(val_steps, val_df["cer"].to_numpy(), 'ro-', linewidth=2, label="CER")
        ax2.plot(val_steps, val_df["wer"].to_numpy(), 'go-', linewidth=2, label="WER")
        ax2.set_xlabel("Validation step")
        ax2.set_ylabel("Error rate")
        ax2.set_title(
            f"Validation CER & WER "
            f"(CER: {val_df['cer'].iloc[-1]:.4f}, WER: {val_df['wer'].iloc[-1]:.4f})"
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig("validation_metrics.png", dpi=300, bbox_inches='tight')
        fig2.savefig("validation_metrics.pdf", bbox_inches='tight')
    else:
        print("No validation data found, skipping validation figure.")
    
    plt.show()
    
    print(f"✓ Saved training_loss.png/pdf and validation_metrics.png/pdf")
    print(f"  Training: {len(train_df)} points, loss {y_train.min():.1f}→{y_train[-1]:.1f}")
    if not val_df.empty:
        print(f"  Final CER: {val_df['cer'].iloc[-1]:.4f}, WER: {val_df['wer'].iloc[-1]:.4f}")

if __name__ == "__main__":
    main()

