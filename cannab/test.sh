mkdir -p foo /wdata/logs

echo "Preparing test files..."
rm /wdata/test_png -r -f
rm /wdata/test_png_5_3_0 -r -f
rm /wdata/test_png_pan_6_7 -r -f
nohup python convert_test.py "$@" > /wdata/logs/convert_test.out &
wait
rm /wdata/pred_50_9ch_fold_* -r -f
echo "Predicting seresnext50 fold 0"
nohup python predict50_9ch_fold.py 0 > /wdata/logs/predict50_0.out &
wait
echo "Predicting seresnext50 fold 1"
nohup python predict50_9ch_fold.py 1 > /wdata/logs/predict50_1.out &
wait
echo "Predicting seresnext50 fold 2"
nohup python predict50_9ch_fold.py 2 > /wdata/logs/predict50_2.out &
wait
echo "Predicting seresnext50 fold 3"
nohup python predict50_9ch_fold.py 3 > /wdata/logs/predict50_3.out &
wait
echo "Predicting seresnext50 fold 4"
nohup python predict50_9ch_fold.py 4 > /wdata/logs/predict50_4.out &
wait
echo "Predicting seresnext50 fold 5"
nohup python predict50_9ch_fold.py 5 > /wdata/logs/predict50_5.out &
wait
echo "Predicting seresnext50 fold 6"
nohup python predict50_9ch_fold.py 6 > /wdata/logs/predict50_6.out &
wait
echo "Predicting seresnext50 fold 7"
nohup python predict50_9ch_fold.py 7 > /wdata/logs/predict50_7.out &
wait
rm /wdata/pred_92_9ch_fold_* -r -f
echo "Predicting dpn92 fold 0"
nohup python predict92_9ch_fold.py 0 > /wdata/logs/predict92_0.out &
wait
echo "Predicting dpn92 fold 1"
nohup python predict92_9ch_fold.py 1 > /wdata/logs/predict92_1.out &
wait
echo "Predicting dpn92 fold 2"
nohup python predict92_9ch_fold.py 2 > /wdata/logs/predict92_2.out &
wait
echo "Predicting dpn92 fold 3"
nohup python predict92_9ch_fold.py 3 > /wdata/logs/predict92_3.out &
wait
echo "Predicting dpn92 fold 4"
nohup python predict92_9ch_fold.py 4 > /wdata/logs/predict92_4.out &
wait
echo "Predicting dpn92 fold 5"
nohup python predict92_9ch_fold.py 5 > /wdata/logs/predict92_5.out &
wait
echo "Predicting dpn92 fold 6"
nohup python predict92_9ch_fold.py 6 > /wdata/logs/predict92_6.out &
wait
echo "Predicting dpn92 fold 7"
nohup python predict92_9ch_fold.py 7 > /wdata/logs/predict92_7.out &
wait
rm /wdata/pred_154_9ch_fold_* -r -f
echo "Predicting senet154 fold 0"
nohup python predict154_9ch_fold.py 0 > /wdata/logs/predict154_0.out &
wait
echo "Predicting senet154 fold 1"
nohup python predict154_9ch_fold.py 1 > /wdata/logs/predict154_1.out &
wait
echo "Predicting senet154 fold 2"
nohup python predict154_9ch_fold.py 2 > /wdata/logs/predict154_2.out &
wait
echo "Predicting senet154 fold 3"
nohup python predict154_9ch_fold.py 3 > /wdata/logs/predict154_3.out &
wait
echo "Predicting senet154 fold 4"
nohup python predict154_9ch_fold.py 4 > /wdata/logs/predict154_4.out &
wait
echo "Predicting senet154 fold 5"
nohup python predict154_9ch_fold.py 5 > /wdata/logs/predict154_5.out &
wait
echo "Predicting senet154 fold 6"
nohup python predict154_9ch_fold.py 6 > /wdata/logs/predict154_6.out &
wait
echo "Predicting senet154 fold 7"
nohup python predict154_9ch_fold.py 7 > /wdata/logs/predict154_7.out &
wait
rm /wdata/pred_101_9ch_fold_* -r -f
echo "Predicting seresnext101 fold 0"
nohup python predict101_9ch_fold.py 0 > /wdata/logs/predict101_0.out &
wait
echo "Predicting seresnext101 fold 1"
nohup python predict101_9ch_fold.py 1 > /wdata/logs/predict101_1.out &
wait
echo "Predicting seresnext101 fold 2"
nohup python predict101_9ch_fold.py 2 > /wdata/logs/predict101_2.out &
wait
echo "Predicting seresnext101 fold 3"
nohup python predict101_9ch_fold.py 3 > /wdata/logs/predict101_3.out &
wait
rm /wdata/merged_pred -r -f
echo "merging predicitons..."
python merge.py
echo "predicting LightGBM"
nohup python predict_classifier.py > /wdata/logs/lgbm_prediction.out &
wait
echo "creating submission"
python create_submission_lgbm.py "$@"