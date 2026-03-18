import os
import re
import argparse
import editdistance
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

def read_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_string_for_wer(s):
    return s.lower().strip()

def extract_page_number(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def compute_all_metrics_per_page(pred_text, gt_text, page_num):
    """Compute ALL OCR evaluation metrics for a single page"""
    metrics = {}
    
    pred_len = len(pred_text)
    gt_len = len(gt_text)
    metrics['gt_chars'] = gt_len
    metrics['pred_chars'] = pred_len
    metrics['char_match_rate'] = pred_len / gt_len if gt_len > 0 else 0
    
    cer_dist = editdistance.eval(pred_text, gt_text)
    metrics['cer'] = cer_dist / gt_len if gt_len > 0 else 1.0
    metrics['cer_raw'] = cer_dist
    metrics['cer_substitutions'] = cer_dist
    
    pred_words = format_string_for_wer(pred_text).split()
    gt_words = format_string_for_wer(gt_text).split()
    metrics['gt_words'] = len(gt_words)
    metrics['pred_words'] = len(pred_words)
    
    wer_dist = editdistance.eval(pred_words, gt_words)
    metrics['wer'] = wer_dist / len(gt_words) if len(gt_words) > 0 else 1.0
    metrics['wer_raw'] = wer_dist
    
    metrics['char_acc'] = 1.0 - metrics['cer']
    metrics['word_acc'] = 1.0 - metrics['wer']
    
    metrics['exact_char_match'] = 1.0 if pred_text == gt_text else 0.0
    metrics['exact_word_match'] = 1.0 if pred_words == gt_words else 0.0
    
    max_len = max(gt_len, pred_len)
    metrics['normed_cer'] = cer_dist / max_len if max_len > 0 else 1.0
    
    pred_word_set = set(pred_words)
    gt_word_set = set(gt_words)
    metrics['word_precision'] = len(pred_word_set & gt_word_set) / len(pred_word_set) if pred_word_set else 0
    metrics['word_recall'] = len(pred_word_set & gt_word_set) / len(gt_word_set) if gt_word_set else 0
    metrics['word_f1'] = 2 * metrics['word_precision'] * metrics['word_recall'] / (metrics['word_precision'] + metrics['word_recall']) if (metrics['word_precision'] + metrics['word_recall']) > 0 else 0
    
    common_chars = sum(min(pred_text.count(c), gt_text.count(c)) for c in set(pred_text + gt_text))
    metrics['char_precision'] = common_chars / pred_len if pred_len > 0 else 0
    metrics['char_recall'] = common_chars / gt_len if gt_len > 0 else 0
    metrics['char_f1'] = 2 * metrics['char_precision'] * metrics['char_recall'] / (metrics['char_precision'] + metrics['char_recall']) if (metrics['char_precision'] + metrics['char_recall']) > 0 else 0
    
    return metrics

def compute_cer_wer_per_page(pred_dir, gt_dir):
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    gt_pattern = r"GT_page_(\d+)\.txt"
    pred_pattern = r"page_(\d+)\.txt"

    gt_map = {extract_page_number(f, gt_pattern): f for f in gt_files if extract_page_number(f, gt_pattern)}
    pred_map = {extract_page_number(f, pred_pattern): f for f in pred_files if extract_page_number(f, pred_pattern)}

    matched_pages = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    
    if not matched_pages:
        print("No matching files found between GT and predictions.")
        return []

    all_page_metrics = []
    
    print(f"\n=== DETAILED METRICS FOR {len(matched_pages)} PAGES ===\n")
    print(f"{'Page':<6} {'CER':<8} {'WER':<8} {'CharAcc':<8} {'WordAcc':<8} {'CharF1':<8} {'NormCER':<8}")
    print("-" * 70)
    
    for page in matched_pages:
        gt_path = os.path.join(gt_dir, gt_map[page])
        pred_path = os.path.join(pred_dir, pred_map[page])

        gt = read_text_file(gt_path)
        pred = read_text_file(pred_path)
        
        metrics = compute_all_metrics_per_page(pred, gt, page)
        metrics['page'] = page
        all_page_metrics.append(metrics)
        
        print(f"{page:<6} {metrics['cer']:<8.4f} {metrics['wer']:<8.4f} {metrics['char_acc']:<8.4f} "
              f"{metrics['word_acc']:<8.4f} {metrics['char_f1']:<8.4f} {metrics['normed_cer']:<8.4f}")

    return all_page_metrics

def statistical_tests(all_page_metrics):
    """Statistical tests on all metrics - STD ONLY"""
    page_cers = [m['cer'] for m in all_page_metrics]
    page_wers = [m['wer'] for m in all_page_metrics]
    pages = [m['page'] for m in all_page_metrics]
    
    print(f"\n=== STATISTICAL SUMMARY FOR {len(pages)} PAGES ===")
    print(f"CER: mean={np.mean(page_cers):.4f}, std={np.std(page_cers):.4f}")
    print(f"WER: mean={np.mean(page_wers):.4f}, std={np.std(page_wers):.4f}")
    
    # Save detailed CSV
    df = pd.DataFrame(all_page_metrics)
    df.to_csv('ocr_all_metrics_per_page.csv', index=False)
    print("\n Detailed metrics saved to 'ocr_all_metrics_per_page.csv'")
    
    return all_page_metrics

def plot_all_metrics(all_page_metrics):
    """Comprehensive visualization - NO BASELINES"""
    pages = [m['page'] for m in all_page_metrics]
    cers = [m['cer'] for m in all_page_metrics]
    wers = [m['wer'] for m in all_page_metrics]
    char_accs = [m['char_acc'] for m in all_page_metrics]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # CER per page
    ax1.bar(pages, cers, alpha=0.7, color='skyblue')
    ax1.axhline(np.mean(cers), color='red', linestyle='--', label=f'Mean: {np.mean(cers):.3f}')
    ax1.set_title('CER per Page')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # WER per page
    ax2.bar(pages, wers, alpha=0.7, color='lightcoral')
    ax2.axhline(np.mean(wers), color='red', linestyle='--', label=f'Mean: {np.mean(wers):.3f}')
    ax2.set_title('WER per Page')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Character Accuracy
    ax3.bar(pages, char_accs, alpha=0.7, color='lightgreen')
    ax3.axhline(np.mean(char_accs), color='red', linestyle='--', label=f'Mean: {np.mean(char_accs):.3f}')
    ax3.set_title('Character Accuracy per Page')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # CER vs WER scatter
    ax4.scatter(cers, wers, s=100, alpha=0.7)
    ax4.set_xlabel('CER')
    ax4.set_ylabel('WER')
    ax4.set_title('CER vs WER Correlation')
    
    plt.tight_layout()
    plt.savefig('ocr_comprehensive_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig("ocr_comprehensive_metrics.pdf", bbox_inches='tight')
    plt.show()
    print("Comprehensive plots saved to 'ocr_comprehensive_metrics.png'")

def main(pred_dir, gt_dir):
    all_page_metrics = compute_cer_wer_per_page(pred_dir, gt_dir)
    stats_results = statistical_tests(all_page_metrics)
    plot_all_metrics(all_page_metrics)
    
    print(f"\n Analysis complete for {len(all_page_metrics)} pages!")
    print("Files generated:")
    print("  - ocr_all_metrics_per_page.csv (ALL metrics per page)")
    print("  - ocr_comprehensive_metrics.png (visualizations)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ALL OCR metrics per page + statistical tests")
    parser.add_argument("predictions_dir", type=str, help="Path to predicted text files")
    parser.add_argument("ground_truth_dir", type=str, help="Ground truth text files")
    
    args = parser.parse_args()
    main(args.predictions_dir, args.ground_truth_dir)
