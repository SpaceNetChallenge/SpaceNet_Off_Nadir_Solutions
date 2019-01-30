import os
from sklearn.model_selection import KFold
import pandas as pd
from params.params import args

if __name__ == '__main__':
	path = os.path.join(args.output_data, 'train', 'images')
	cv_total = args.n_folds
	seed = args.seed

	all_files = os.listdir(path)
	all_ids = list(set(['_'.join(el.split('_')[4:]).split('.')[0] for el in all_files]))
	all_nadirs = list(set(['_'.join(el.split('_')[:4]) for el in all_files]))

	target_df = {'img_id': [], 'fold_on_train': [], 'fold_on_predict': []}
	target_df['img_id'] += all_ids
	target_df['fold_on_train'] += [-1 for el in all_ids]
	target_df['fold_on_predict'] += [-1 for el in all_ids]
	target_df = pd.DataFrame(target_df)

	kf = KFold(n_splits=cv_total, random_state=seed, shuffle=True)
	for i, (train_index, evaluate_index) in enumerate(kf.split(target_df.index.values)):
	    target_df['fold_on_predict'].iloc[evaluate_index] = i
	    target_df['fold_on_train'].iloc[evaluate_index[:30]] = i
	target_df = target_df[['img_id', 'fold_on_train', 'fold_on_predict']]

	final = {'img_id': [], 'fold_on_train': [], 'fold_on_predict': []}
	for row in target_df.values:
	    for nadir in all_nadirs:
	        img_id = '_'.join([nadir, row[0]])
	        final['img_id'].append(img_id)
	        final['fold_on_train'].append(row[1])
	        final['fold_on_predict'].append(row[2])
	        # print(img_id)
	final = pd.DataFrame(final)
	final.to_csv(args.folds_file, index=False)