"""
Script for misc resources used in this script broadly.
"""


### LISTS AND DICTIONARIES ###
lang_LH_rois = ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp', 'lang_LH_netw', ]

d_roi_lists_names = {'lang_LH_netw': ['lang_LH_netw'],
					 'lang_RH_netw': ['lang_RH_netw'],
					 'lang_LH_IFGorb': ['lang_LH_IFGorb'],
					 'lang_RH_IFGorb': ['lang_RH_IFGorb'],
					 'lang_LH_IFG': ['lang_LH_IFG'],
					 'lang_RH_IFG': ['lang_RH_IFG'],
					 'lang_LH_MFG': ['lang_LH_MFG'],
					 'lang_RH_MFG': ['lang_RH_MFG'],
					 'lang_LH_AntTemp': ['lang_LH_AntTemp'],
					 'lang_RH_AntTemp': ['lang_RH_AntTemp'],
					 'lang_LH_PostTemp': ['lang_LH_PostTemp'],
					 'lang_RH_PostTemp': ['lang_RH_PostTemp'],

					 'md_LH_netw': ['md_LH_netw'],
					 'md_RH_netw': ['md_RH_netw'],
					 'dmn_LH_netw': ['dmn_LH_netw'],
					 'dmn_RH_netw': ['dmn_RH_netw'],
					'lang_LH_ROIs': ['lang_LH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp', ],
					 'lang_RH_ROIs': ['lang_RH_netw', 'lang_RH_IFGorb', 'lang_RH_IFG', 'lang_RH_MFG', 'lang_RH_AntTemp', 'lang_RH_PostTemp', ],
					 'lang_LHRH_ROIs': ['lang_LH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp',
										'lang_RH_netw', 'lang_RH_IFGorb', 'lang_RH_IFG', 'lang_RH_MFG', 'lang_RH_AntTemp', 'lang_RH_PostTemp', ],
					 'audvis_anatglasser_ROIs': ['anatglasser_LH_AudPrimary', 'anatglasser_RH_AudPrimary',
									   'anatglasser_LH_V1', 'anatglasser_RH_V1',],
					 # 'lang_anatglasser_ROIs': list(d_anatglasser_networks['lang'].keys()),
					 # 'anatglasser_LH_LangNetw': ['anatglasser_LH_LangNetw'],
					 # 'lang_LH_anatglasser_ROIs': [x for x in list(d_anatglasser_networks['lang'].keys()) if x.startswith('anatglasser_LH_') and not x.endswith('AngG')],
					 # 'lang_RH_anatglasser_ROIs': [x for x in list(d_anatglasser_networks['lang'].keys()) if x.startswith('anatglasser_RH_') and not x.endswith('AngG')],

					 # 'lang_LH_normal_anatglasser_ROIs': ['lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp', 'lang_LH_netw',
						# 								 'anatglasser_LH_LangIFGorb', 'anatglasser_LH_LangIFG', 'anatglasser_LH_LangMFG', 'anatglasser_LH_LangAntTemp', 'anatglasser_LH_LangPostTemp', 'anatglasser_LH_LangNetw'],
					 # 'all_ROIs': lst_all_rois,
					 # 'rois_func_lang_md_dmn': rois_func_lang_md_dmn,
					 }

rois_func_lang_md_dmn = [
	'lang_LH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp',
	'lang_RH_netw', 'lang_RH_IFGorb', 'lang_RH_IFG', 'lang_RH_MFG', 'lang_RH_AntTemp', 'lang_RH_PostTemp',
	'md_LH_netw', 'md_LH_postParietal', 'md_LH_midParietal', 'md_LH_antParietal', 'md_LH_supFrontal',
	'md_LH_PrecentralAprecG', 'md_LH_PrecentralBIFGop', 'md_LH_midFrontal', 'md_LH_midFrontalOrb', 'md_LH_insula',
	'md_LH_medialFrontal',
	'md_RH_netw', 'md_RH_postParietal', 'md_RH_midParietal', 'md_RH_antParietal', 'md_RH_supFrontal',
	'md_RH_PrecentralAprecG', 'md_RH_PrecentralBIFGop', 'md_RH_midFrontal', 'md_RH_midFrontalOrb', 'md_RH_insula',
	'md_RH_medialFrontal',
	'dmn_LH_netw', 'dmn_LH_FrontalMed', 'dmn_LH_PostCing', 'dmn_LH_TPJ', 'dmn_LH_MidCing', 'dmn_LH_STGorInsula',
	'dmn_LH_AntTemp',
	'dmn_RH_netw', 'dmn_RH_FrontalMed', 'dmn_RH_PostCing', 'dmn_RH_TPJ', 'dmn_RH_MidCing', 'dmn_RH_STGorInsula',
	'dmn_RH_AntTemp', ]

d_colors = {'S': 'royalblue',
			'D': 'tab:red',
			'B': 'grey',
			'D_search': 'tab:red',
			'S_search': 'royalblue',
			 'D_modify': 'tab:red',
			 'S_modify': 'royalblue',
			'lang': 'lightcoral',
			'md': 'royalblue',
			'dmn': 'mediumseagreen',
			'vis': 'orange',
			'aud': 'darkorchid'
			}


d_netw_colors = {
	'lang_LH_netw': 'firebrick',
	'lang_RH_netw': 'firebrick',
	'lang_LHRH_netw': 'firebrick',
	'lang_LH_IFGorb': 'firebrick',
	'lang_LH_IFG': 'firebrick',
	'lang_LH_MFG': 'firebrick',
	'lang_LH_AntTemp': 'firebrick',
	'lang_LH_PostTemp': 'firebrick',
	'lang_LH_AngG': 'firebrick',
	'lang_RH_IFGorb': 'firebrick',
	'lang_RH_IFG': 'firebrick',
	'lang_RH_MFG': 'firebrick',
	'lang_RH_AntTemp': 'firebrick',
	'lang_RH_PostTemp': 'firebrick',
	'lang_RH_AngG': 'firebrick',

	'md_LH_netw': 'mediumblue',
	'md_RH_netw': 'mediumblue',
	'md_LHRH_netw': 'mediumblue',
	'md_LH_PrecentralAprecG': 'mediumblue',
	'md_LH_PrecentralBIFGop': 'mediumblue',
	'md_LH_antParietal': 'mediumblue',
	'md_LH_insula': 'mediumblue',
	'md_LH_medialFrontal': 'mediumblue',
	'md_LH_midFrontal': 'mediumblue',
	'md_LH_midParietal': 'mediumblue',
	'md_LH_postParietal': 'mediumblue',
	'md_LH_supFrontal': 'mediumblue',
	'md_RH_PrecentralAprecG': 'mediumblue',
	'md_RH_PrecentralBIFGop': 'mediumblue',
	'md_RH_antParietal': 'mediumblue',
	'md_RH_insula': 'mediumblue',
	'md_RH_medialFrontal': 'mediumblue',
	'md_RH_midFrontal': 'mediumblue',
	'md_RH_midFrontalOrb': 'mediumblue',
	'md_RH_midParietal': 'mediumblue',
	'md_RH_postParietal': 'mediumblue',
	'md_RH_supFrontal': 'mediumblue',

	'dmn_LH_netw': 'forestgreen',
	'dmn_RH_netw': 'forestgreen',
	'dmn_LHRH_netw': 'forestgreen',
	'dmn_LH_AntTemp': 'forestgreen',
	'dmn_LH_FrontalMed': 'forestgreen',
	'dmn_LH_MidCing': 'forestgreen',
	'dmn_LH_PostCing': 'forestgreen',
	'dmn_LH_STGorInsula': 'forestgreen',
	'dmn_LH_TPJ': 'forestgreen',
	'dmn_RH_AntTemp': 'forestgreen',
	'dmn_RH_FrontalMed': 'forestgreen',
	'dmn_RH_MidCing': 'forestgreen',
	'dmn_RH_PostCing': 'forestgreen',
	'dmn_RH_STGorInsula': 'forestgreen',
	'dmn_RH_TPJ': 'forestgreen',

	# 'anatglasser_LH_V1': 'sienna',
	# 'anatglasser_RH_V1': 'sienna',
	# 'anatglasser_LHRH_V1': 'sienna',
	#
	# 'anatglasser_LH_AudPrimary': 'slategrey',
	# 'anatglasser_RH_AudPrimary': 'slategrey',
	# 'anatglasser_LHRH_AudPrimary': 'slategrey',
}


d_symbols = {'B': 'o',
			 'D_search': 's',
			 'D_modify': '^',
			 'S_search': 's',
			 'S_modify': '^',}


d_axes_legend = {'CV_score_mean': 'Cross-validated predictivity (mean ± fold SE)',
				  'CV_score_median': 'Cross-validated predictivity (median ± fold SE)',
				 'CV_alpha_mean': 'Cross-validated mean alpha',
				 'CV_alpha_median': 'Cross-validated median alpha',
				 'full_score': 'Held-out participant predictivity',
				 'cond': 'Condition',
				 'response_target': 'Z-scored BOLD response (mean)',
				 'response_target_non_norm': 'Non-normalized BOLD response (mean)',
				 'response': 'BOLD response',
				 'Actual (797-841-880-837-856-848-853-865-875-876)': 'BOLD response (mean)',
				 'std_over_items': 'item SD',
				  'sem_over_items': 'item SE',
				'std_over_UIDs':'participant SD',
				'sem_over_UIDs': 'participant SE',
										 'std_within_UIDs': 'within-participant SD',
					 'sem_within_UIDs': 'within-participant SE',
				 'encoding_model_pred': 'Encoding model prediction',
				 'pred-CV-k-5_from-848-853-865-875-876': 'Predicted from 848-853-865-875-876 (CV)',
				 'pretransformer_pred-surprisal-gpt2-xl-surprisal-gpt2-xl_mean': 'Prediction from "GPT2-XL surprisal model"',
				 'pretransformer_pred-surprisal-5gram-surprisal-5gram_mean': 'Prediction from "5-gram surprisal model"',
				 'pretransformer_pred-surprisal-pcfg-surprisal-pcfg_mean': 'Prediction from "PCFG surprisal model"',
				 'encoding_model_pred_noise': 'Simulated BOLD response (mean)',
				 'pretransformer_pred_noise': 'Pretransformer pred NOISE',
				 'nc': 'Noise ceiling (± split-half SE)',
				 'surprisal-gpt2-xl_mean': 'Surprisal',  #(GPT2-XL)
				 'surprisal-gpt2-xl_raw_mean': 'Log probability',
				 'log-prob-gpt2-xl_mean': 'Log probability',
				 'surprisal-gpt2-xl_sum': 'Surprisal (sum)',
				 'surprisal-5gram_mean': 'Surprisal (5-gram)',
				 'surprisal-5gram_raw_mean': 'Log probability (5-gram)',
				 'surprisal-5gram_sum': 'Surprisal (5-gram, sum)',
				'surprisal-pcfg_mean': 'Surprisal (PCFG)',
				'surprisal-pcfg_raw_mean': 'Log probability (PCFG)',
				'surprisal-pcfg_sum': 'Surprisal (PCFG, sum)',
				 'rating_arousal_mean': 'Arousal',
				 'rating_conversational_mean': 'Conversational frequency',
				 'rating_sense_mean': 'Plausibility',
				 'rating_gram_mean': 'Grammaticality',
				 'rating_frequency_mean': 'General frequency',
				 'rating_imageability_mean': 'Imageability',
				 'rating_memorability_mean': 'Memorability',
				 'rating_others_thoughts_mean': 'Mental states',
				 'rating_physical_mean': 'Physical objects',
				 'rating_places_mean': 'Places',
				 'rating_valence_mean': 'Valence',

				 'D_search': 'Search',
				 'D_modify': 'Modify',
				 'S_search': 'Search',
				 'S_modify': 'Modify',
				 'B': 'Baseline',

				 'T_1gram_overlap': '1-gram overlap',
				 'T_2gram_overlap': '2-gram overlap',
				 'T_3gram_overlap': '3-gram overlap',
				 'T_4gram_overlap': '4-gram overlap',
				 'T_5gram_overlap': '5-gram overlap',
				 'T_6gram_overlap': '6-gram overlap',
				 }


### FUNCTIONS ###
def shorten_savestr(savestr: str):
	"""
	Replace True and False with 1 and 0.
	Replace None with 0.
	Replace 'anatglasser' with 'ag'
	"""
	savestr = savestr.replace('True', '1').replace('False', '0').replace('None', '0').replace('anatglasser', 'ag')

	# Also shorten 848-853-865-875-876 to 5T and 797-841-880-837-856 to 5D
	savestr = savestr.replace('848-853-865-875-876', '5T').replace('797-841-880-837-856', '5D')

	# Shorten the feature stimset from 'beta-control-neural_stimset_D-S_light_compiled-feats' to 'bcn_D-S_feats
	savestr = savestr.replace('beta-control-neural_stimset_D-S_light_compiled-feats', 'bcn_D-S_feats')

	# Shorten bert-large-cased to bert-lc
	savestr = savestr.replace('bert-large-cased', 'bert-lc')

	# Shorten surprisal-gpt2-xl_raw_mean to log_prob-gpt2-xl
	savestr = savestr.replace('surprisal-gpt2-xl_raw_mean', 'log_prob-gpt2-xl')

	# And also shorten the other features
	savestr = savestr.replace('rating_gram_mean', 'gram')
	savestr = savestr.replace('rating_sense_mean', 'sense')
	savestr = savestr.replace('rating_others_thoughts_mean', 'others')
	savestr = savestr.replace('rating_physical_mean', 'physical')
	savestr = savestr.replace('rating_places_mean', 'places')
	savestr = savestr.replace('rating_valence_mean', 'valence')
	savestr = savestr.replace('rating_arousal_mean', 'arousal')
	savestr = savestr.replace('rating_imageability_mean', 'imageability')
	savestr = savestr.replace('rating_frequency_mean', 'frequency')
	savestr = savestr.replace('rating_conversational_mean', 'conversational')

	# For pretransformer predictions
	# pretransformer_pred-surprisal-gpt2-xl-surprisal-gpt2-xl_mean to pred-surp-gpt2-xl_mean
	savestr = savestr.replace('pretransformer_pred-surprisal-gpt2-xl-surprisal-gpt2-xl_mean', 'pred-surp-gpt2-xl_mean')
	savestr = savestr.replace('pretransformer_pred-surprisal-5gram-surprisal-5gram_mean', 'pred-surp-5gram_mean')
	savestr = savestr.replace('pretransformer_pred-surprisal-pcfg-surprisal-pcfg_mean', 'pred-surp-pcfg_mean')

	return savestr

def item_scatter_style(style_setting: str):
	"""
	Available style settings for the item scatter: 'square', 'wide'

	"""

	if style_setting == 'square':
		plot_aspect_flag = 1
		add_identity_flag = True

		d_xlim = {'lang_LH_netw':
					  {'None':
						   {'797-841-880-837-856': None, },
					   'bySessVoxZ':
						   {'797-841-880-837-856': None,
							'797-841-880': [-1.7, 1.7],
							'837-856': [-2.4, 2.4],
							'797': None,
							'841': None,
							'880': None,
							'837': None,
							'856': None,
							}}}
		d_ylim = d_xlim

	elif style_setting == 'wide':
		plot_aspect_flag = 0.8
		add_identity_flag = True

		d_xlim = {'lang_LH_netw':
					  {'None':
						   {'797-841-880-837-856': None,
							'797-841-880': [-0.75, 0.75],
							'837-856': [-1, 1],  # synth has different preds
							'797': [-0.75, 0.75],
							'841': [-0.75, 0.75],
							'880': [-0.75, 0.75],
							'837': [-1, 1],
							'856': [-1, 1], },
					   'bySessVoxZ':
						   {'797-841-880-837-856': None,
							'797-841-880': [-0.75, 0.75],
							'837-856': [-1, 1],  # synth has different preds
							'797': [-0.75, 0.75],
							'841': [-0.75, 0.75],
							'880': [-0.75, 0.75],
							'837': [-1, 1],
							'856': [-1, 1],
							}}}
		d_ylim = {'lang_LH_netw':
					  {'None':
						   {'797-841-880-837-856': None,
							'797-841-880': [-2, 2],
							'837-856': [-2.4, 2.4],
							'797': None,
							'841': None,
							'880': None,
							'837': None,
							'856': None, },
					   'bySessVoxZ':
						   {'797-841-880-837-856': None,
							'797-841-880': [-2, 2],
							'837-856': [-2.4, 2.4],
							'797': [-3, 3],
							'841': [-3, 3],
							'880': [-3, 3],
							'837': [-3.5, 3.5],
							'856': [-3.5, 3.5],
							}}}
	else:
		raise ValueError(f'Invalid style_setting: {style_setting}')

	return plot_aspect_flag, add_identity_flag, d_xlim, d_ylim