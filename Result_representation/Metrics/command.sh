python crop_line.py  --xml 44/page_44.xml --images 44/ --output 44/lines



python compute_cer_wer.py ./predictions ./groundtruth


To compare CER and WER for all images run this command :

python compare_pages_cer_wer.py our_model/ GT/


To plot the training process :

python train_progress_figure.py run.log


To compare between 3 method :

python compute_all_metrics_molinoDS.py bvmp/ pero-ocr/ our-model/ GT/
