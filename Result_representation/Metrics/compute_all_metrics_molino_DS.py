import os
import re
import argparse
import editdistance
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def read_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_string_for_wer(s):
    return s.lower().strip()

def extract_page_number(filename, pattern):
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def compute_cer_wer_single(pred_dir, gt_dir, method_name):
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))
    
    gt_pattern = r"GT_page_(\d+)\.txt"
    pred_pattern = r"page_(\d+)\.txt"
    
    gt_map = {extract_page_number(f, gt_pattern): f for f in gt_files if extract_page_number(f, gt_pattern)}
    pred_map = {extract_page_number(f, pred_pattern): f for f in pred_files if extract_page_number(f, pred_pattern)}
    
    matched_pages = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    page_metrics = []
    total_cer, total_wer, len_gt_cer, len_gt_wer = 0, 0, 0, 0
    
    for page in matched_pages:
        gt_path = os.path.join(gt_dir, gt_map[page])
        pred_path = os.path.join(pred_dir, pred_map[page])
        
        gt = read_text_file(gt_path)
        pred = read_text_file(pred_path)
        
        cer_dist = editdistance.eval(pred, gt)
        cer_page = cer_dist / len(gt) if len(gt) > 0 else 1.0
        total_cer += cer_dist
        len_gt_cer += len(gt)
        
        pred_words = format_string_for_wer(pred).split()
        gt_words = format_string_for_wer(gt).split()
        wer_dist = editdistance.eval(pred_words, gt_words)
        wer_page = wer_dist / len(gt_words) if len(gt_words) > 0 else 1.0
        total_wer += wer_dist
        len_gt_wer += len(gt_words)
        
        page_metrics.append({
            'page': page, 'method': method_name,
            'cer': cer_page, 'wer': wer_page
        })
    
    cer_global = total_cer / len_gt_cer if len_gt_cer > 0 else float('inf')
    wer_global = total_wer / len_gt_wer if len_gt_wer > 0 else float('inf')
    
    return {
        'method': method_name,
        'page_metrics': page_metrics,
        'cer_global': cer_global,
        'wer_global': wer_global
    }

def compare_three_methods(pred_dir1, pred_dir2, pred_dir3, gt_dir):
    method_names = ["ABBY", "PERO-OCR", "FP-THD"]
    
    results = []
    results.append(compute_cer_wer_single(pred_dir1, gt_dir, method_names[0]))
    results.append(compute_cer_wer_single(pred_dir2, gt_dir, method_names[1]))
    results.append(compute_cer_wer_single(pred_dir3, gt_dir, method_names[2]))
    
    all_page_data = []
    for r in results:
        all_page_data.extend(r['page_metrics'])
    
    df = pd.DataFrame(all_page_data)
    df.to_csv('ocr_comparison.csv', index=False)
    
    print(f"\n{'='*120}")
    print(f"{'OCR PERFORMANCE (Molino Dataset, n=10 pages/method)':^120}")
    print(f"{'Method':<12} {'CER':<10} {'WER':<10} {'CER-μ':<10} {'WER-μ':<10} {'σ_CER':<10} {'σ_WER':<10} {'p_CER':<12} {'p_WER':<12}")
    print("-" * 120)
    
    for r in results:
        page_cers = np.array([m['cer'] for m in r['page_metrics']])
        page_wers = np.array([m['wer'] for m in r['page_metrics']])
        
        cer_mean = np.mean(page_cers)
        cer_std = np.std(page_cers, ddof=1)
        wer_mean = np.mean(page_wers)
        wer_std = np.std(page_wers, ddof=1)
        
        cer_p = stats.ttest_1samp(page_cers, 0.10).pvalue
        wer_p = stats.ttest_1samp(page_wers, 0.20).pvalue
        
        print(f"{r['method']:<12} {r['cer_global']:<10.4f} {r['wer_global']:<10.4f} "
              f"{cer_mean:<10.4f} {wer_mean:<10.4f} {cer_std:<10.4f} {wer_std:<10.4f} "
              f"{cer_p:<12.1e} {wer_p:<12.1e}")
    
    best_method = min(results, key=lambda x: x['cer_global'])
    print(f"\n BEST: {best_method['method']} (CER = {best_method['cer_global']:.1%})")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    methods = [r['method'] for r in results]
    x = np.arange(len(methods))
    
    cer_globals = [r['cer_global'] for r in results]
    wer_globals = [r['wer_global'] for r in results]
    
    width = 0.35
    ax.bar(x - width/2, cer_globals, width, label='CER', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, wer_globals, width, label='WER', alpha=0.8, color='coral')
    ax.axhline(0.10, color='red', linestyle='--', label='CER baseline 10%')
    ax.axhline(0.20, color='orange', linestyle='--', label='WER baseline 20%')
    ax.set_title('OCR Comparison (ABBY vs PERO-OCR vs FP-THD)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ocr_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig("ocr_comparison.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"\n SAVED: ocr_comparison.csv + ocr_comparison.png")
    return results, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Comparison")
    parser.add_argument("pred_dir1", type=str, help="ABBY predictions")
    parser.add_argument("pred_dir2", type=str, help="PERO-OCR predictions")
    parser.add_argument("pred_dir3", type=str, help="FP-THD predictions")
    parser.add_argument("gt_dir", type=str, help="Ground truth")
    
    args = parser.parse_args()
    results, df = compare_three_methods(
        args.pred_dir1, args.pred_dir2, args.pred_dir3, args.gt_dir
    )

