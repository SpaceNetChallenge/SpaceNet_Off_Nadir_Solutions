mkdir -p foo /wdata/logs

echo "Creating masks..."
nohup python create_masks.py "$@" > /wdata/logs/create_masks.out &
wait
echo "Masks created"

echo "training seresnext50 folds 0-3"
nohup python train50_9ch_fold.py 0 > /wdata/logs/train50_0.out &
nohup python train50_9ch_fold.py 1 > /wdata/logs/train50_1.out &
nohup python train50_9ch_fold.py 2 > /wdata/logs/train50_2.out &
nohup python train50_9ch_fold.py 3 > /wdata/logs/train50_3.out &
wait

echo "training seresnext50 folds 4-7"
nohup python train50_9ch_fold.py 4 > /wdata/logs/train50_4.out &
nohup python train50_9ch_fold.py 5 > /wdata/logs/train50_5.out &
nohup python train50_9ch_fold.py 6 > /wdata/logs/train50_6.out &
nohup python train50_9ch_fold.py 7 > /wdata/logs/train50_7.out &
wait

echo "training dpn92 folds 0-3"
nohup python train92_9ch_fold.py 0 > /wdata/logs/train92_0.out &
nohup python train92_9ch_fold.py 1 > /wdata/logs/train92_1.out &
nohup python train92_9ch_fold.py 2 > /wdata/logs/train92_2.out &
nohup python train92_9ch_fold.py 3 > /wdata/logs/train92_3.out &
wait

echo "training dpn92 folds 4-7"
nohup python train92_9ch_fold.py 4 > /wdata/logs/train92_4.out &
nohup python train92_9ch_fold.py 5 > /wdata/logs/train92_5.out &
nohup python train92_9ch_fold.py 6 > /wdata/logs/train92_6.out &
nohup python train92_9ch_fold.py 7 > /wdata/logs/train92_7.out &
wait

echo "training senet154 folds 0-3"
nohup python train154_9ch_fold.py 0 > /wdata/logs/train154_0.out &
nohup python train154_9ch_fold.py 1 > /wdata/logs/train154_1.out &
nohup python train154_9ch_fold.py 2 > /wdata/logs/train154_2.out &
nohup python train154_9ch_fold.py 3 > /wdata/logs/train154_3.out &
wait

echo "training senet154 folds 4-7"
nohup python train154_9ch_fold.py 4 > /wdata/logs/train154_4.out &
nohup python train154_9ch_fold.py 5 > /wdata/logs/train154_5.out &
nohup python train154_9ch_fold.py 6 > /wdata/logs/train154_6.out &
nohup python train154_9ch_fold.py 7 > /wdata/logs/train154_7.out &
wait

echo "training seresnext101 folds 0-3"
nohup python train101_9ch_fold.py 0 > /wdata/logs/train101_0.out &
nohup python train101_9ch_fold.py 1 > /wdata/logs/train101_1.out &
nohup python train101_9ch_fold.py 2 > /wdata/logs/train101_2.out &
nohup python train101_9ch_fold.py 3 > /wdata/logs/train101_3.out &
wait
echo "All NNs trained!"

rm /wdata/pred_50_9ch_oof_0 -r -f
echo "predicting seresnext50 out-of-fold 0-3"
nohup python predict50_9ch_oof.py 0 > /wdata/logs/oof50_0.out &
nohup python predict50_9ch_oof.py 1 > /wdata/logs/oof50_1.out &
nohup python predict50_9ch_oof.py 2 > /wdata/logs/oof50_2.out &
nohup python predict50_9ch_oof.py 3 > /wdata/logs/oof50_3.out &
wait

echo "predicting seresnext50 out-of-fold 4-7"
nohup python predict50_9ch_oof.py 4 > /wdata/logs/oof50_4.out &
nohup python predict50_9ch_oof.py 5 > /wdata/logs/oof50_5.out &
nohup python predict50_9ch_oof.py 6 > /wdata/logs/oof50_6.out &
nohup python predict50_9ch_oof.py 7 > /wdata/logs/oof50_7.out &
wait

rm /wdata/pred_92_9ch_oof_0 -r -f
echo "predicting dpn92 out-of-fold 0-3"
nohup python predict92_9ch_oof.py 0 > /wdata/logs/oof92_0.out &
nohup python predict92_9ch_oof.py 1 > /wdata/logs/oof92_1.out &
nohup python predict92_9ch_oof.py 2 > /wdata/logs/oof92_2.out &
nohup python predict92_9ch_oof.py 3 > /wdata/logs/oof92_3.out &
wait

echo "predicting dpn92 out-of-fold 4-7"
nohup python predict92_9ch_oof.py 4 > /wdata/logs/oof92_4.out &
nohup python predict92_9ch_oof.py 5 > /wdata/logs/oof92_5.out &
nohup python predict92_9ch_oof.py 6 > /wdata/logs/oof92_6.out &
nohup python predict92_9ch_oof.py 7 > /wdata/logs/oof92_7.out &
wait

rm /wdata/pred_154_9ch_oof_0 -r -f
echo "predicting senet154 out-of-fold 0-3"
nohup python predict154_9ch_oof.py 0 > /wdata/logs/oof154_0.out &
nohup python predict154_9ch_oof.py 1 > /wdata/logs/oof154_1.out &
nohup python predict154_9ch_oof.py 2 > /wdata/logs/oof154_2.out &
nohup python predict154_9ch_oof.py 3 > /wdata/logs/oof154_3.out &
wait

echo "predicting senet154 out-of-fold 4-7"
nohup python predict154_9ch_oof.py 4 > /wdata/logs/oof154_4.out &
nohup python predict154_9ch_oof.py 5 > /wdata/logs/oof154_5.out &
nohup python predict154_9ch_oof.py 6 > /wdata/logs/oof154_6.out &
nohup python predict154_9ch_oof.py 7 > /wdata/logs/oof154_7.out &
wait

rm /wdata/pred_101_9ch_oof_0 -r -f
echo "predicting seresnext101 out-of-fold 0-3"
nohup python predict101_9ch_oof.py 0 > /wdata/logs/oof101_0.out &
nohup python predict101_9ch_oof.py 1 > /wdata/logs/oof101_1.out &
nohup python predict101_9ch_oof.py 2 > /wdata/logs/oof101_2.out &
nohup python predict101_9ch_oof.py 3 > /wdata/logs/oof101_3.out &
wait

rm /wdata/merged_oof -r -f
echo "merging oof predicitons..."
python merge_oof.py

echo "training LightGBM models..."
nohup python train_classifier.py > /wdata/logs/lgbm_train.out &
wait

echo "All models trained!"