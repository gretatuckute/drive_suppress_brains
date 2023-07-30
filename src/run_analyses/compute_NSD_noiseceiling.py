"""
Script for computing the NSD noise ceiling for subsets of subjects, across ROIs.

"""
import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import NSD_noiseceiling
from src.paths import RESULTROOT, DATAROOT
from datetime import datetime
date = datetime.now().strftime("%Y%m%d-%T")

script_name = os.path.basename(__file__).split('.')[0]
PLOTDIR = join(RESULTROOT, 'plots', script_name)
CSVDIR = join(RESULTROOT, 'csvs', script_name)
if not os.path.exists(PLOTDIR):
	os.makedirs(PLOTDIR)
	print(f'Created PLOTDIR: {PLOTDIR}')
if not os.path.exists(CSVDIR):
	os.makedirs(CSVDIR)
	print(f'Created CSVDIR: {CSVDIR}')

##### SETTINGS #####
UIDs =  [848, 853, 865, 875, 876]
UID_str = '-'.join([str(x) for x in UIDs])
target_dataset_fname = 'brain-lang-data_participant_20230728.csv'

rois = ['lang_LH_netw']

normal_NC_estimation = True # no bootstrap
split_half_NC_estimation = False # estimate error on the noise ceiling via bootstrap across sentences
bootstrap = False
save = False

#### Compute "true" noise ceiling ####
if normal_NC_estimation:
	NC_n = 5 # If None, use n_UIDs

	lst_across_rois = [] # Store noise ceiling across ROIs
	##### ROI LOOP #####
	for roi in rois:

		fname = f'{DATAROOT}/{target_dataset_fname}'
		df = pd.read_csv(fname)

		# Take the 5 train participants
		df_train = df.query('target_UID in @UIDs')
		df_train_roi = df_train.query('roi == @roi')

		# Make into a num_stim x 5 UIDs matrix
		df_across_UIDs = df_train_roi.pivot(index='item_id', columns='target_UID', values='response_target')

		n_items = df_across_UIDs.shape[0]
		n_UIDs = df_across_UIDs.shape[1]

		temp = df_across_UIDs.values  # (1000, n_UIDs)

		# Compute noise ceiling
		# If NC_n is not specified, use n_UIDs
		NC_n = n_UIDs if NC_n is None else NC_n

		noiseceiling, ncsnr, sd_signal, sd_noise = NSD_noiseceiling(data=temp,
																	NC_n=NC_n) # For non-bootstrap, use the n_UIDs as the number of samples that are averaged together to compute the noise ceiling

		# Package into df
		df = pd.DataFrame({'noiseceiling': noiseceiling,
						   'ncsnr': ncsnr,
						   'sd_signal': sd_signal,
						   'sd_noise': sd_noise,
						   'roi': roi,
						   'n_items': n_items,
						   'n_UIDs': n_UIDs,
						   'NC_n': NC_n,
						   'UIDs': [UIDs],
						   'UID_str': UID_str,
						   }, index=[0])
		df.index = [roi]

		lst_across_rois.append(df)

	##### END ROI LOOP #####

	df_across_rois = pd.concat(lst_across_rois, axis=0)

	if save:
		save_str = f'n-rois-{len(rois)}_NC_n-{NC_n}'

		# Log
		df_across_rois['save_str'] = save_str
		df_across_rois['rois'] = rois
		df_across_rois[f'date_{script_name}'] = date

		df_across_rois.to_csv(join(CSVDIR, script_name, save_str + '.csv'))


#### Compute split half noise ceiling ####
if split_half_NC_estimation:
	NC_n = 5 # If None, use n_UIDs
	n_splits = 1000
	plot_distribution = False

	lst_across_rois = [] # Store noise ceiling across ROIs

	##### ROI LOOP #####
	for roi in rois:
		print(f'\nComputing noise ceiling for {roi}...')
		fname = f'{DATAROOT}/{target_dataset_fname}'
		df = pd.read_csv(fname)

		# Take the 5 train participants
		df_train = df.query('target_UID in [848, 853, 865, 875, 876]')
		# Make into a num_stim x 5 UIDs matrix
		df_across_UIDs = df_train.pivot(index='item_id', columns='target_UID', values='response_target')

		n_items = df_across_UIDs.shape[0]
		n_UIDs = df_across_UIDs.shape[1]

		# If NC_n is not specified, use n_UIDs
		NC_n = n_UIDs if NC_n is None else NC_n

		##### SPLIT HALF LOOP #####
		lst_rand_splits = []
		for split_idx in range(n_splits):

			# Reset random seed such that the same UIDs are drawn for each ROI (makes the noise ceiling comparable across ROIs)
			np.random.seed(split_idx)

			# Split df_across_UIDs into two halves (half of items, all UIDs)
			rand_idx_s1 = np.random.choice(n_items, size=int(n_items/2), replace=False)
			rand_idx_s2 = np.setdiff1d(np.arange(n_items), rand_idx_s1)
			df_across_UIDs_s1 = df_across_UIDs.iloc[rand_idx_s1, :]
			df_across_UIDs_s2 = df_across_UIDs.iloc[rand_idx_s2, :]

			## Assertions
			# Check that the idxs are unique
			assert len(np.unique(np.concatenate((rand_idx_s1, rand_idx_s2)))) == n_items
			# Get items
			n_items_in_s1 = df_across_UIDs_s1.shape[0]
			n_items_in_s2 = df_across_UIDs_s2.shape[0]
			assert n_items_in_s1 == n_items_in_s2
			# Check that indices are not the same
			items_in_s1 = df_across_UIDs_s1.index
			items_in_s2 = df_across_UIDs_s2.index
			assert items_in_s1.equals(items_in_s2) == False

			temp_s1 = df_across_UIDs_s1.values # (n_items / 2, n_UIDs)
			temp_s2 = df_across_UIDs_s2.values # (n_items / 2, n_UIDs)

			# Compute noise ceiling
			# If NC_n is not specified, use n_UIDs
			NC_n = n_UIDs if NC_n is None else NC_n

			noiseceiling_s1, ncsnr_s1, sd_signal_s1, sd_noise_s1 = NSD_noiseceiling(data=temp_s1,
																		NC_n=NC_n)
			noiseceiling_s2, ncsnr_s2, sd_signal_s2, sd_noise_s2 = NSD_noiseceiling(data=temp_s2,
																		NC_n=NC_n)

			# Compute std
			split_half_std_per_split = (np.std([noiseceiling_s1, noiseceiling_s2], ddof=1)) # Even if N=2, we still want the unbiased estimator
			split_half_se_per_split = split_half_std_per_split / np.sqrt(2)

			# Package into df
			df_split_half = pd.DataFrame({'split_noiseceiling_s1': noiseceiling_s1,
										  'split_ncsnr_s1': ncsnr_s1,
										  'split_sd_signal_s1': sd_signal_s1,
										  'split_sd_noise_s1': sd_noise_s1,
										  'split_noiseceiling_s2': noiseceiling_s2,
										  'split_ncsnr_s2': ncsnr_s2,
										  'split_sd_signal_s2': sd_signal_s2,
										  'split_sd_noise_s2': sd_noise_s2,
										  'split_half_std_per_split': split_half_std_per_split, # this is the std value computed over the two independent NC estimates
										  'split_half_se_per_split': split_half_se_per_split,
										  'split_idx': split_idx,
										  'split_n_items': n_items_in_s1,
										  'split_roi': roi,
										  }, index=[0])
			df_split_half.index = [f'{roi}_{split_idx}']

			if roi == 'lang_LH_netw': # Add in the item_id values for each split. If we add in item_ids across all ROIs, the df size blows up
				df_split_half['split_item_ids_in_s1'] = [items_in_s1.values] # df_across_uids is returned with the item_id number as index
				df_split_half['split_item_ids_in_s2'] = [items_in_s2.values]

			lst_rand_splits.append(df_split_half)

		##### END SPLIT HALF LOOP #####
		# Concatenate across split halves
		df_split_half_roi = pd.concat(lst_rand_splits, axis=0)
		lst_across_rois.append(df_split_half_roi)

	##### END ROI LOOP #####
	# Concatenate across ROIs
	df_split_half = pd.concat(lst_across_rois, axis=0)

	### Compute mean and SE across split halves ###
	# For each ROI, concatenate split_noiseceiling_s1 and split_noiseceiling_s2
	for roi in df_split_half['split_roi'].unique():
		df_roi = df_split_half[df_split_half['split_roi'] == roi]

		# We want to pool across all the SD values obtained across the 1,000 split halves for each ROI
		# To do so, we square the individual SD values → take the mean → sqrt
		split_half_std_mean = np.sqrt(np.mean(df_roi['split_half_std_per_split']**2)) # We want to average the variances

		# Divide by sqrt(2) to get SE
		split_half_se = split_half_std_mean / np.sqrt(2) # division by number of splits, i.e. N_split = 2

		# Compute other statistics across split halves
		noiseceiling_s1 = df_roi['split_noiseceiling_s1'].values
		noiseceiling_s2 = df_roi['split_noiseceiling_s2'].values
		noiseceiling_concat = np.concatenate((noiseceiling_s1, noiseceiling_s2))
		# Compute mean
		noiseceiling_concat_mean = np.mean(noiseceiling_concat)
		# Compute std
		noiseceiling_concat_std = np.std(noiseceiling_concat, ddof=1)
		# Compute 68% CI
		noiseceiling_concat_ci_68 = np.percentile(noiseceiling_concat, [16, 84])
		# Compute 95% CI
		noiseceiling_concat_ci_95 = np.percentile(noiseceiling_concat, [2.5, 97.5])

		# Add to df
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_half_std_mean'] = split_half_std_mean
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_half_se'] = split_half_se # This is our split-half SE of interest

		# Additional misc stats
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_noiseceiling_concat_mean'] = noiseceiling_concat_mean
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_noiseceiling_concat_std'] = noiseceiling_concat_std
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_noiseceiling_concat_ci_16'] = noiseceiling_concat_ci_68[0]
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_noiseceiling_concat_ci_84'] = noiseceiling_concat_ci_68[1]
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_noiseceiling_concat_ci_2.5'] = noiseceiling_concat_ci_95[0]
		df_split_half.loc[df_split_half['split_roi'] == roi, 'split_noiseceiling_concat_ci_97.5'] = noiseceiling_concat_ci_95[1]


	if plot_distribution:
		for roi in df_split_half['split_roi'].unique():
			df_roi = df_split_half[df_split_half['split_roi'] == roi]

			# Plot distrib of noiseceiling_s1 and noiseceiling_s2
			fig, ax = plt.subplots(1, 1, figsize=(7, 7))
			ax.hist(df_roi['split_noiseceiling_s1'], bins=20, alpha=0.5, label='noiseceiling_s1')
			ax.hist(df_roi['split_noiseceiling_s2'], bins=20, alpha=0.5, label='noiseceiling_s2')
			ax.set_xlabel('NC_Pearson')
			ax.set_ylabel('Count')
			ax.set_title(f'Split-half NC_Pearson for {roi} across {n_splits} splits')
			ax.legend()
			plt.tight_layout()
			plt.show()

			# Show distrib of split_half_se
			fig, ax = plt.subplots(1, 1, figsize=(5, 5))
			ax.hist(df_roi['split_half_std_per_split'], bins=20, alpha=0.5, label='split_half_std_per_split')
			ax.set_xlabel('split_half_std_per_split')
			ax.set_ylabel('Count')
			ax.set_title(f'Split-half SE for {roi} across {n_splits} splits')
			ax.legend()
			plt.tight_layout()
			plt.show()


	### Add more metadata to df_split_half
	df_split_half['split_n_UIDs'] = n_UIDs
	df_split_half['split_UIDs'] = [UIDs] * len(df_split_half)
	df_split_half['split_UID_str'] = UID_str
	df_split_half[f'date_{script_name}'] = date


	if save:
		save_str =  f'n-rois-{len(rois)}_' \
				   f'n-splits-{n_splits}_' \
				   f'NC_n-{NC_n}'

		# Log
		df_split_half['save_str'] = save_str
		df_split_half['n_rois'] = len(rois)

		df_split_half.to_csv(join(CSVDIR, script_name, save_str + '.csv'))
