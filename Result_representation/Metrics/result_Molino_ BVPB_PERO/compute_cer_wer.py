import os
import argparse
import editdistance


def read_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_string_for_wer(s):
    return s.lower().strip()


def compute_cer_wer(pred_dir, gt_dir):
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    total_cer = 0
    total_wer = 0
    len_gt_cer = 0
    len_gt_wer = 0

    for filename in pred_files:
        pred_path = os.path.join(pred_dir, filename)
        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            print(f"Warning: Missing ground truth for '{filename}', skipping.")
            continue

        pred = read_text_file(pred_path)
        gt = read_text_file(gt_path)

        # CER
        cer_distance = editdistance.eval(pred, gt)
        total_cer += cer_distance
        len_gt_cer += len(gt)

        # WER
        pred_words = format_string_for_wer(pred).split()
        gt_words = format_string_for_wer(gt).split()
        wer_distance = editdistance.eval(pred_words, gt_words)
        total_wer += wer_distance
        len_gt_wer += len(gt_words)

    cer = total_cer / len_gt_cer if len_gt_cer > 0 else float('inf')
    wer = total_wer / len_gt_wer if len_gt_wer > 0 else float('inf')

    print(f"\nResults:")
    print(f"CER (Character Error Rate): {cer:.4f}")
    print(f"WER (Word Error Rate):      {wer:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CER and WER between prediction and ground truth text files.")
    parser.add_argument("predictions_dir", type=str, help="Path to the folder containing predicted text files.")
    parser.add_argument("ground_truth_dir", type=str, help="Path to the folder containing ground truth text files.")

    args = parser.parse_args()
    compute_cer_wer(args.predictions_dir, args.ground_truth_dir)
