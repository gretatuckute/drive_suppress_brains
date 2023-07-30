from BrainClasses import *
from src.paths import RESULTROOT, WEIGHTROOT, DATAROOT, ACTVROOT, LOGROOT
script_name = os.path.basename(__file__).split('.')[0]

"""
Script for loading fitted, stored mapping weights (from fit_mapping) and perform predictions on any stimuli set WITHOUT
brain responses.

Allows to:
- Store predictions for the supplied stimset.

"""

def main(raw_args=None):
	parser = argparse.ArgumentParser(description='')
	
	## Shared parameters among weights and brain target ##
	# ANN (source) specific (used for loading weights and computing embeddings for the new target stimset)
	parser.add_argument('--source_model', default='gpt2-xl', type=str, help='Pretrained model name')
	parser.add_argument('--source_layer', default=22, help='Which layer to use for prediction')
	parser.add_argument('--sent_embed', default='last-tok', type=str, help='How to obtain sentence embeddings')
	parser.add_argument('--actv_cache_setting', default='auto', type=str, help='Which cache setting to use')
	
	# Mapping specific
	parser.add_argument('--mapping_class', default='ridgeCV', type=str, help='Which mapping class to use')
	parser.add_argument('--metric', default='pearsonr', type=str, help='Which metric to use')
	parser.add_argument('--preprocessor', default='None', type=str2none, help='How to preprocess data prior to mapping. Options are "None", "demean", "demean_std", "minmax"')
	parser.add_argument('--preprocess_X', default=False, type=str2bool, help='Whether to preprocess X (regressors) prior to mapping. Must specify preprocessor if True.')
	parser.add_argument('--preprocess_y', default=False, type=str2bool, help='Whether to preprocess y (targets) prior to mapping. Must specify preprocessor if True.')
	parser.add_argument('--store_full_pred', default=True, type=str2bool, help='Whether to store predictions on the full target dataset')
	parser.add_argument('--permute_X', default=None, type=str2none, help='Whether to permute X (regressors) prior to mapping'
																		'Options are None, "shuffle_X_cols", "shuffle_X_rows", "shuffle_each_X_col"')

	# Mapping (weight) specific
	parser.add_argument('--mapping_UID', default='848-853-865-875-876', type=str, help='Unique subject ID')
	parser.add_argument('--mapping_sess_num', default='1-2', type=str, help='Which session(s) to load data for')
	parser.add_argument('--mapping_FL', default='gs', type=str, help='First level (FL) identifier')
	parser.add_argument('--mapping_modeltype', default='d', type=str, help='Which GLMsingle model type to load')
	parser.add_argument('--mapping_preproc', default='swr', type=str, help='Input data to GLMsingle')
	parser.add_argument('--mapping_pcstop', default=5, type=int, help='Num of PCs removed in GLMsingle')
	parser.add_argument('--mapping_fracs', default=0.05, type=float, help='Ridge regression fraction')
	parser.add_argument('--mapping_func_thresh', default=90, type=int, help='Threshold for extracting functional ROIs')
	parser.add_argument('--mapping_norm', default='bySessVoxZ', type=str, help='Which normalization to use (while extracting and packaging ROIs)')
	parser.add_argument('--mapping_regression_dict_type', default='rois', type=str, help='Whether to load "rois" or "voxs" neural data dicts')
	parser.add_argument('--mapping_savestr_prefix', default='20221214a', type=str, help='If the netw_dict was stored using a savestr_prefix, specify it here.')
	parser.add_argument('--mapping_specific_target', default=None, type=str2none, help='Whether to run a specific target')

	
	# Corpus (stimset) specific
	parser.add_argument('--stimset_path', default='/om2/user/gretatu/script_repos/beta-neural-control/material_selection/drive_set_selection_synth/',
						type=str, help='Path directory of the stimset')
	parser.add_argument('--stimset_filename', default='beta-control-neural_stimset_D.csv', type=str, help='Filename of the stimset')
	parser.add_argument('--stimset_colname', default='sentence', type=str, help='Name of the column in the stimset that contains the sentences')
	parser.add_argument('--compute_stretch_metrics', default=False, type=str2bool, help='Whether to compute stretch metrics for the predicted '
																					   'responses of the stimset. Stretch metrics are based on the'
																					   'neural data that the mapping was derived from.')


	# Misc
	parser.add_argument('--verbose', default=True, type=str2bool, help='Whether to print output and not create a log file')
	
	####### Arguments and logging #######
	args = parser.parse_args(raw_args)
	print(vars(args)) # To add it to the .out file prior logging
	
	args_logger = ArgumentLogger(vars(args),
								 script_name=script_name,
								 add_args={'mapping_sess_id': args.mapping_sess_num,
										   'script_name': script_name,
										   f'date_{script_name}': date},
								 result_root=RESULTROOT,
								 weight_root=WEIGHTROOT,
								 log_root=LOGROOT,
								 actv_root=ACTVROOT,
								 )
	args_logger.create_save_str()

		
	####### CREATE LOGGING FILE WITH CORRECT PARAMETERS ########
	if not args.verbose:
		logfile = join(args_logger.LOGDIR, f"{script_name}_{args_logger.save_str}_{date}.log")
		# If path does not exist, create it.
		os.makedirs("/".join(logfile.split('/')[:-1]), exist_ok=True)
		print(f'\nLogging output to file...\n {logfile}')
		sys.stdout = open(logfile, 'a+')
	
	print('\n' + ('*' * 40))
	print(vars(args))
	print(('*' * 40) + '\n')
	args_logger.print_package_versions()
	
	### Obtain mapping (weight) folder (result_identifier) and mapping (weight) save string (save_str)
	mapping_result_identifier = f'SOURCE-{args.source_model}_' \
								f'{args.sent_embed}_' \
								f'TARGET-{args.mapping_regression_dict_type}-' \
								f'{args.mapping_UID}_' \
								f'{args.mapping_sess_num}_' \
								f'{args.mapping_FL}_' \
								f'{args.mapping_func_thresh}_' \
								f'MAPPING-{args.mapping_class}-' \
								f'{args.metric}'

	mapping_save_str = f'SOURCE-{args.source_layer}_' \
					   f'TARGET-{args.mapping_savestr_prefix}-' \
					   f'{args.mapping_specific_target}_' \
					   f'{args.mapping_modeltype}-' \
					   f'{args.mapping_preproc}-' \
					   f'{args.mapping_pcstop}-' \
					   f'{args.mapping_fracs}-' \
					   f'{args.mapping_norm}_' \
					   f'MAPPING-{args.preprocessor}-' \
					   f'{args.preprocess_X}-' \
					   f'{args.preprocess_y}'
	
	args_logger.add_key_val({'mapping_result_identifier': mapping_result_identifier,
							 'mapping_save_str': mapping_save_str})


	####### "BRAIN" (CORPUS) ENCODER ########
	stimset = pd.read_csv(join(args.stimset_path,
							   args.stimset_filename), index_col=0)
	
	brain = BrainEncoder()
	brain.encode(stimset=stimset,
				 neural_data=None,
				 specific_target=None)
	sys.stdout.flush()

	####### ANN ENCODER ########
	ann = ANNEncoder(source_model=args.source_model,
					 sent_embed=args.sent_embed,
					 actv_cache_setting=args.actv_cache_setting,
					 actv_cache_path=args_logger.ACTVDIR)
	
	# Encode the new brain target stimset
	ann.encode(stimset=stimset,
			   stim_col=args.stimset_colname,
			   cache_new_actv=True,
			   case=None,
			   **{'stimsetid_suffix': f''},
			   include_special_tokens=True,
			   verbose=False)

	####### METRIC ########
	metric = Metric(metric=args.metric)
	
	####### PREPROCESSOR ########
	preprocessor = Preprocessor(preprocess=args.preprocessor)
	
	####### MAPPING ########
	mapping = Mapping(ANNEncoder=ann,
					  ann_layer=args.source_layer,
					  BrainEncoder=brain,
					  mapping_class=args.mapping_class,
					  metric=metric,
					  Preprocessor=preprocessor,
					  preprocess_X=args.preprocess_X,
					  preprocess_y=args.preprocess_y,)
	
	#### Load stored mapping weights #####
	mapping.load_full_mapping(WEIGHTDIR=WEIGHTROOT,
							  mapping_result_identifier=mapping_result_identifier,
							  mapping_save_str=f'mapping-full_{mapping_save_str}.pkl')
	
	#### Check the mapping (weights) against the information that was stored in the CV results df ####
	check_mapping_against_CV_results = True
	if check_mapping_against_CV_results:
		df_cv_scores = pd.read_pickle(join(RESULTROOT,
										   'fit_mapping',
										   mapping_result_identifier,
										   f'CV-k-5_{mapping_save_str}.pkl'))
		
		assert (df_cv_scores.index.values == mapping.prefitted_clf_neuroid_order).all()
		assert (df_cv_scores.full_alpha == mapping.prefitted_clf.alpha_).all()
		assert (df_cv_scores.source_model == args.source_model).all()
		# Make sure that source layer is int in both cases
		df_cv_scores_source_layer = df_cv_scores.source_layer.astype(int)
		args_source_layer = int(args.source_layer)
		assert (df_cv_scores_source_layer == args_source_layer).all()
		assert (df_cv_scores.UID == args.mapping_UID).all()
		assert (df_cv_scores.mapping_class == args.mapping_class).all()
		assert (df_cv_scores.result_identifier == mapping_result_identifier).all()
		assert (df_cv_scores.save_str == mapping_save_str).all()
		print(f'== Passed mapping weights vs CV results check for {mapping_result_identifier}')
		
	#### Predict the stimset ####
	df_preds = mapping.predict_using_prefitted_mapping()
	
	
	if args.compute_stretch_metrics:
		# Load the neural values that were used to obtain the mapping weights (for computing stretch metrics)
		pred_full = pd.read_pickle(join(RESULTROOT,
										   'fit_mapping',
										   mapping_result_identifier,
										   f'pred-full_{mapping_save_str}.pkl'))
	else:
		pred_full = None
	
	#### Package the results #####
	df_preds_packaged = package_pred_results_as_df(df=df_preds,
												   stimset=stimset,
												   stimset_cols_to_drop=['pred', 'roi', 'y_max', 'y_min',
												 'y_pred_max', 'y_pred_min', 'stretch_pred_max', 'stretch_pred_min',
												 'stretch_pred_max_minmax', 'stretch_pred_min_minmax', 'stretch_max',
												 'stretch_min', 'stretch_max_minmax', 'stretch_min_minmax',
												 'source_model', 'source_layer', 'sent_embed', 'actv_cache_setting',
												 'mapping_class', 'metric', 'preprocess_X', 'preprocess_y',
												 'store_full_pred', 'mapping_UID', 'mapping_sess_num', 'mapping_FL',
												 'mapping_modeltype', 'mapping_preproc', 'mapping_pcstop',
												 'mapping_fracs', 'mapping_func_thresh', 'mapping_norm', 'ACTVROOT',
												 'corpus_filename', 'top_bottom_n_stim', 'verbose', 'result_identifier',
												 'mapping_sess_id', 'save_str', 'mapping_result_identifier',
												 'mapping_save_str', ],
												 pred_full=pred_full,)
	
	# If CV results were loaded from the mapping model, add in the CV results to the packaged results
	if check_mapping_against_CV_results:
		df_preds_packaged = add_CV_results_to_df(df_preds_packaged=df_preds_packaged,
												   df_cv_scores=df_cv_scores)
		
	
	############## ADD OTHER METADATA ##############
	
	if args.store_full_pred:
		df_preds_packaged = args_logger.add_args_to_df(df=df_preds_packaged)
		
		# Check for duplicated column names
		assert df_preds_packaged.columns.duplicated().sum() == 0
		
		args_logger.store(data=df_preds_packaged, # Only store the predictions because real brain data does not exist
						  DIR = 'RESULTDIR',
						  prefix_str='pred-full',)
	
	sys.stdout.flush()


if __name__ == '__main__':
	main()


