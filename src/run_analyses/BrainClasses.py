from utils import *

torch.set_default_dtype(torch.double)


class ANNEncoder:
	def __init__(self, source_model: str = 'gpt2',
				 sent_embed: str = 'last-tok',
				 # these are used to define the cache, thus make them a part of the constructor
				 actv_cache_setting: typing.Union[str, None] = 'auto',  # if auto, use cache. If None, don't use cache
				 actv_cache_path: typing.Union[str, None] = None) -> None:

		self._source_model = source_model

		if source_model.startswith('gpt') or source_model.startswith('bert'):
			# Pretrained model
			from transformers import AutoModel, AutoConfig, AutoTokenizer
			self.config = AutoConfig.from_pretrained(self._source_model)
			self.tokenizer = AutoTokenizer.from_pretrained(self._source_model)
			self.model = AutoModel.from_pretrained(self._source_model, config=self.config)
		self.sent_embed = sent_embed

		# Cache
		self.user = getpass.getuser()
		self.actv_cache_setting = actv_cache_setting
		self.actv_cache_path = actv_cache_path

	def _aggregate_layers(self, hidden_states: dict,
						  sent_embed: str = 'last-tok') -> None:
		"""[summary]
		Args:
			hidden_states (torch.Tensor): pytorch tensor of shape (n_items, dims)
			sent_embed: an object specifying the method to use for aggregating
													representations across items within a layer
		Raises:
			NotImplementedError
		Returns:
			np.ndarray: the aggregated array
		"""
		states_layers = dict()
		for i in hidden_states.keys():  # for each layer
			if sent_embed == 'last-tok':
				state = hidden_states[i][-1, :]  # get last token
			elif sent_embed == 'first-tok':
				state = hidden_states[i][0, :]  # get first token
			elif sent_embed == 'mean-tok':
				state = torch.mean(hidden_states[i], dim=0)  # mean over tokens
			elif sent_embed == 'median-tok':
				state = torch.median(hidden_states[i], dim=0)  # median over tokens
			elif sent_embed == 'sum-tok':
				state = torch.sum(hidden_states[i], dim=0)  # sum over tokens
			elif sent_embed == 'all-tok' or sent_embed == None:
				state = hidden_states
			else:
				raise NotImplementedError('Sentence embedding method not implemented')

			states_layers[i] = state.detach().numpy()

		return states_layers

	def _flatten_activations(self, states_sentences_agg: dict,
							 index: str = 'DEFAULTINDEX'):
		"""Flatten activations.

		Args:
			states_sentences_agg (dict): dictionary of activations (key: layer, value: activations as ndarray)
			index (str, optional): index to use for flattening (usually the stimid index). Defaults to 'DEFAULTINDEX'.

		Returns:
			df (pandas.DataFrame): Multi-indexed dataframe of flattened activations.
			Rows are sentences (indexed by index), columns are units flattened across layers.
			The first column index is the layer, the second is the unit.
		"""

		labels = []
		lst_arr_flat = []
		for layer, arr in states_sentences_agg.items():
			arr = np.array(arr)  # for each layer
			lst_arr_flat.append(arr)
			# Create multiindex for each layer. index 0 is the layer index, and index 1 is the unit index
			for i in range(arr.shape[0]):  # across units
				labels.append((layer, i))
		arr_flat = np.concatenate(lst_arr_flat)  # concatenated activations across layers
		df = pd.DataFrame(arr_flat).T
		df.index = [index]
		df.columns = pd.MultiIndex.from_tuples(labels)  # rows: stimuli, columns: units
		return df

	def _create_actv_cache_path(self, ):
		os.makedirs(self.actv_cache_path, exist_ok=True)

	def _case(self, sample: str = None,
			  case: typing.Union[str, None] = None):
		if case == 'lower':
			sample = sample.lower()
		elif case == 'upper':
			sample = sample.upper()
		else:
			sample = sample

		return sample

	def get_special_token_offset(self) -> int:
		'''
		the offset (no. of tokens in tokenized text) from the start to exclude
		when extracting the representation of a particular stimulus. this is
		needed when the stimulus is evaluated in a context group to achieve
		correct boundaries (otherwise we get off-by-context errors)
		'''
		with_special_tokens = self.tokenizer("brainscore")['input_ids']
		first_token_id, *_ = self.tokenizer("brainscore", add_special_tokens=False)['input_ids']
		special_token_offset = with_special_tokens.index(first_token_id)
		return special_token_offset

	def get_context_groups(self,
						   stimset: pd.DataFrame = None,
						   context_dim: typing.Union[str, None] = None, ):
		""""Initialize the context group coordinate (obtain embeddings with context)"""
		if context_dim is None:
			context_groups = np.arange(0, len(stimset), 1)
		else:
			context_groups = stimset[context_dim].values

		return context_groups


	def encode_from_csv(self,
					   stimset: pd.DataFrame = None,
					   stim_col: str = 'sentence',
					   cache_new_actv: bool = True,
						CSVDIR: str = None,
						CSV_fname: str = None,
						**kwargs):
		"""
		Load activations from a csv file.

		:param stimset:
		:param stim_col:
		:param cache_new_actv:
		:param verbose:
		:param CSVDIR:
		:param kwargs:
		:return:
		"""

		# Obtain stimsetid (the identifier for the stimuli)
		stimsetid_all = ['.'.join(stimset.index[x].split('.')[:-1]) for x in
						 range(len(stimset))]  # include all information separated by '.' besides the very last index
		assert (len(np.unique(stimsetid_all)) == 1)  # Check whether all sentences come from the same corpus
		stimsetid = stimsetid_all[0]
		if kwargs.get('stimsetid_suffix'):
			stimsetid_suffix = kwargs.get('stimsetid_suffix')
			# Add "_" to the end of the stimsetid if it doesn't already have one
			if stimsetid_suffix[0] != '_':
				stimsetid_suffix = '_' + stimsetid_suffix
			stimsetid = f'{stimsetid}{stimsetid_suffix}'

		self.stimset = stimset
		self.stimsetid = stimsetid
		self.stim_col = stim_col

		stim_fname = f'{self.stimsetid}_stim.pkl'
		actv_fname = f'{self.stimsetid}_actv.pkl'

		### Check if we have already computed activations for this corpus (stimsetid) ###
		if self.actv_cache_setting == 'auto':
			self._create_actv_cache_path()
			stim_fname = f'{self.actv_cache_path}/{stim_fname}'
			actv_fname = f'{self.actv_cache_path}/{actv_fname}'
			if os.path.exists(f'{actv_fname}'):
				print(f'Loading cached ANN encoder activations for {self.stimsetid} from {self.actv_cache_path}\n')
				stim = pd.read_pickle(stim_fname)
				actv = pd.read_pickle(actv_fname)
				assert (self.stimset.index == stim.index).all()
				assert (actv.index == stim.index).all()
				self.encoded_ann = actv
				return self.encoded_ann

		### If not, load the activations from a csv file ###
		if CSVDIR is None:
			raise ValueError('Please specify the directory of the csv file.')

		# Load the csv file
		df_csv = pd.read_csv(join(CSVDIR,
								  self._source_model,
								  f'{CSV_fname}.csv'), index_col=0)

		if self._source_model == 'surprisal-pcfg':
			df_csv['item_id'] = df_csv.index
			df_csv = df_csv.set_index('stimsetid', drop=True)

		# If the index of the df_csv is 'beta-control-neural-D.{int}', we want to replace it with with 'beta-control-neural-T.{int}' in df_csv
		if df_csv.index[0].startswith('beta-control-neural-D') and stimsetid.startswith('beta-control-neural-T'):
			df_csv.index = df_csv.index.str.replace('beta-control-neural-D', 'beta-control-neural-T')
			print(f'Replaced "beta-control-neural-D" with "beta-control-neural-T" in df_csv.index\n')

		# Only get the indices that are indices in stimset
		df_csv_encoded = df_csv.loc[stimset.index]

		# Perform assertions between stimset and df_csv
		assert(stimset.item_id.values == df_csv_encoded.item_id.values).all() # Assumes that we have item_id in stimset
		assert(stimset[self.stim_col].values == df_csv_encoded['sentence'].values).all()

		# Obtain the activations in the column of interest (self.sent_embed)
		actv = df_csv_encoded[[self.sent_embed]]

		# Add multiindex 0 in columns in actv (for consistency with other encoders that have multiple layers)
		actv.columns = pd.MultiIndex.from_product([[0], actv.columns])

		print(f'Number of stimuli in activations: {actv.shape[0]}\n'
			  f'Number of units in activations: {actv.shape[1]}\n')

		assert (stimset.index == actv.index).all()

		if cache_new_actv:
			stimset.to_pickle(stim_fname, protocol=4)
			actv.to_pickle(actv_fname, protocol=4)
			print(f'\nCaching newly computed activations!\nCached activations to {actv_fname}')

		self.encoded_ann = actv
		self.encoded_stimset = stimset

		return self.encoded_ann

	def encode(self,
			   stimset: pd.DataFrame = None,
			   stim_col: str = 'sentence',
			   case: typing.Union[str, None] = None,
			   context_dim: str = None,
			   bidirectional: bool = False,
			   include_special_tokens: bool = True,
			   cache_new_actv: bool = True,
			   verbose: bool = False,
			   **kwargs):
		""" Input a pandas dataframe with stimuli, encode and return a pandas dataframe with activations.
  
		Args:
			stimset (pd.DataFrame): a pandas dataframe with stimuli. If caching is enabled, the index has to adhere to 'stimid.0'
			stim_col (str): the column in the dataframe with the stimuli
			case (str): the case to use for the stimuli
			context_dim (str): the dimension to use for the context groupings of sampleids (stimuli) that should be used
												as context when generating encoder representations.
												If None, then no context is used.
			bidirectional (bool): if True, allows using "future" context to generate the representation for a current token
											otherwise, only uses what occurs in the "past". some might say, setting this to False
											gives you a more biologically plausibly analysis downstream.
			include_special_tokens (bool): if True, includes the special tokens in the representation (e.g. [CLS] and [SEP]).
							Note that the sequence is always tokenized using using add_special_tokens=True, so the special tokens
							are sliced out in the embeddings if this is set to False. E.g. if the sequence is tokenized as
							[CLS] this is a sentence [SEP], then the representation returned with include_special_tokens=False
							will be the representation of "this is a sentence", while if include_special_tokens=True, the
							representation will be the representation of [CLS] this is a sentence [SEP].

			cache_new_actv (bool): if True, caches the activations to self.actv_cache_path (a file with stimuli (suffix _stim)
															and activations (suffix _actv) is created) as pkl files.
		Returns:
			Sets self.encoded_ann to a pandas dataframe with activations.
			Sets self.encoded_stimset to a pandas dataframe with stimuli.
		"""
		# Obtain stimsetid (the identifier for the stimuli)
		stimsetid_all = ['.'.join(stimset.index[x].split('.')[:-1]) for x in
						 range(len(stimset))]  # include all information separated by '.' besides the very last index
		assert (len(np.unique(stimsetid_all)) == 1)  # Check whether all sentences come from the same corpus
		stimsetid = stimsetid_all[0]
		if kwargs.get('stimsetid_suffix'):
			stimsetid_suffix = kwargs.get('stimsetid_suffix')
			# Add "_" to the end of the stimsetid if it doesn't already have one
			if stimsetid_suffix[0] != '_':
				stimsetid_suffix = '_' + stimsetid_suffix
			stimsetid = f'{stimsetid}{stimsetid_suffix}'

		self.stimset = stimset
		self.stimsetid = stimsetid
		self.stim_col = stim_col

		stim_fname = f'{self.stimsetid}_stim.pkl'
		actv_fname = f'{self.stimsetid}_actv.pkl'

		### Check if we have already computed activations for this corpus (stimsetid) ###
		if self.actv_cache_setting == 'auto':
			self._create_actv_cache_path()
			stim_fname = f'{self.actv_cache_path}/{stim_fname}'
			actv_fname = f'{self.actv_cache_path}/{actv_fname}'
			if os.path.exists(f'{actv_fname}'):
				print(f'Loading cached ANN encoder activations for {self.stimsetid} from {self.actv_cache_path}\n')
				stim = pd.read_pickle(stim_fname)
				actv = pd.read_pickle(actv_fname)
				assert (self.stimset.index == stim.index).all()
				assert (actv.index == stim.index).all()
				self.encoded_ann = actv
				return self.encoded_ann

		self.model.eval()
		stimuli = self.stimset[self.stim_col].values

		# Initialize the context group coordinate (obtain embeddings with context)
		context_groups = self.get_context_groups(stimset=stimset, context_dim=context_dim)

		###############################################################################
		# ALL SAMPLES LOOP
		###############################################################################
		states_sentences_across_groups = []
		stim_index_counter = 0
		_, unique_ixs = np.unique(context_groups, return_index=True)
		for group in tqdm(context_groups[np.sort(unique_ixs)]):  # Make sure context group order is preserved
			mask_context = context_groups == group
			stim_in_context = stimuli[mask_context]  # Mask based on the context group

			states_sentences_across_stim = []  # Store states for each sample in this context group

			###############################################################################
			# CONTEXT LOOP
			###############################################################################
			for i, stimulus in enumerate(stim_in_context):
				stimulus = self._case(sample=stimulus, case=case)

				if len(stim_in_context) > 1:
					print(f'encoding stimulus {i} of {len(stim_in_context)}')

				# mask based on the uni/bi-directional nature of models :)
				if not bidirectional:
					stim_directional = stim_in_context[: i + 1]
				else:
					stim_directional = stim_in_context

				# join the stimuli together within a context group using just a single space
				stim_directional = " ".join(stim_directional)

				stim_directional = self._case(sample=stim_directional, case=case)

				tokenized_directional_context = self.tokenizer(stim_directional,
															   padding=False,
															   return_tensors='pt',
															   add_special_tokens=True)

				# Get the hidden states
				result_model = self.model(tokenized_directional_context.input_ids,
										  output_hidden_states=True,
										  return_dict=True)
				hidden_states = result_model[
					'hidden_states']  # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]

				layerwise_activations = defaultdict(list)

				# Find which indices match the current stimulus in the given context group
				start_of_interest = stim_directional.find(stimulus)
				char_span_of_interest = slice(
					start_of_interest, start_of_interest + len(stimulus)
				)
				token_span_of_interest = pick_matching_token_ixs(
					tokenized_directional_context, char_span_of_interest
				)

				if verbose:
					print(f'Interested in the following stimulus:\n{stim_directional[char_span_of_interest]}\n'
						  f'Recovered:\n{tokenized_directional_context.tokens()[token_span_of_interest]}')  # See which tokens are used (with the special tokens)

				all_special_ids = set(self.tokenizer.all_special_ids)

				# Look for special tokens in the beginning and end of the sequence
				insert_first_upto = 0
				insert_last_from = tokenized_directional_context.input_ids.shape[-1]
				# loop through input ids
				for i, tid in enumerate(tokenized_directional_context.input_ids[0, :]):
					if tid.item() in all_special_ids:
						insert_first_upto = i + 1
					else:
						break
				for i in range(1, tokenized_directional_context.input_ids.shape[-1] + 1):
					tid = tokenized_directional_context.input_ids[0, -i]
					if tid.item() in all_special_ids:
						insert_last_from -= 1
					else:
						break

				for idx_layer, layer in enumerate(hidden_states):  # Iterate over layers
					# b (1), n (tokens), h (768, ...)
					# collapse batch dim to obtain shape (n_tokens, emb_dim)
					this_extracted = layer[
									 :,
									 token_span_of_interest,
									 :,
									 ].squeeze(0)

					if include_special_tokens:  # This will concatenate the obtained embeddings (obtained once, together with the stimulus)
						# with the embeddings of the special tokens (i.e., "layer" and "this_extracted" will be the same if include_special_tokens=True)
						# get the embeddings for the first special tokens
						this_extracted = torch.cat(
							[
								layer[:, :insert_first_upto, :].squeeze(0),
								this_extracted,
							],
							axis=0,
						)
						# get the embeddings for the last special tokens
						this_extracted = torch.cat(
							[
								this_extracted,
								layer[:, insert_last_from:, :].squeeze(0),
							],
							axis=0,
						)

					layerwise_activations[idx_layer] = this_extracted.detach()

				# aggregate within a stimulus
				states_sentences_agg = self._aggregate_layers(layerwise_activations,
															  sent_embed=self.sent_embed)
				# dict with key=layer, value=array of # size [emb dim]

				# Convert to flattened pandas df
				current_stimid = stimset.index[stim_index_counter]
				assert (self._case(sample=stimset.loc[current_stimid][stim_col], case=case) == stimulus)

				df_states_sentences_agg = self._flatten_activations(states_sentences_agg,
																	index=current_stimid)

				# append the dfs to states_sentences_across_stim (which is ALL stim within a context group)
				states_sentences_across_stim.append(df_states_sentences_agg)
				# now we have all the hidden states for the current context group across all stimuli

				stim_index_counter += 1

			###############################################################################
			# END CONTEXT LOOP
			###############################################################################

			states_sentences_across_groups.append(pd.concat(states_sentences_across_stim, axis=0))

		###############################################################################
		# END ALL SAMPLES LOOP
		###############################################################################

		actv = pd.concat(states_sentences_across_groups, axis=0)

		print(f'Number of stimuli in activations: {actv.shape[0]}\n'
			  f'Number of units in activations: {actv.shape[1]}\n')

		assert (stimset.index == actv.index).all()

		if cache_new_actv:
			stimset.to_pickle(stim_fname, protocol=4)
			actv.to_pickle(actv_fname, protocol=4)
			print(f'\nCaching newly computed activations!\nCached activations to {actv_fname}')

		self.encoded_ann = actv
		self.encoded_stimset = stimset

		return self.encoded_ann


class Metric:
	def __init__(self,
				 metric: str = 'pearsonr',
				 rsm_metric: str = 'pearsonr',):
		self.metric_name = metric # For comparing e.g. ANN and brain
		self.rsm_metric = rsm_metric # For computing RSMs

	def _check_neuroids(self,
						A: typing.Any = None,
						B: typing.Any = None
						):
		"""Assert that the two matrices to be compared have same number of neuroids

		Args
			:arg A (typing.Union[pd.DataFrame, np.ndarray]) [description] Expected shape is [number of data points; neuroids]
					Number of data points is how many points will be compared  for each column (i.e. neuroid)
			:arg B

		"""
		assert (A.shape == B.shape)
		num_data_points = A.shape[0]
		num_neuroids = A.shape[1]

		print(
			f'Comparing similarity using {self.metric_name} for {num_neuroids} neuroids, {num_data_points} data points each\n')

		return num_neuroids

	def _index_into_df_or_ndarray(self, A: typing.Union[pd.DataFrame, np.ndarray],
								  idx: typing.Union[int, typing.List[int]]):
		"""Index into the columns of either a dataframe or ndarray"""
		if type(A) == pd.DataFrame:
			A_indexed = A.iloc[:, idx]
		elif type(A) == np.ndarray:
			A_indexed = A[:, idx]
		else:
			raise TypeError(f'A is of type {type(A)}')

		return A_indexed

	def _metric_over_neuroids(self,
							  A: typing.Union[pd.DataFrame, np.ndarray] = None,
							  B: typing.Union[pd.DataFrame, np.ndarray] = None):
		"""Evalutes the chosen metric (similarity) per neuroid, i.e. per column in the supplied matrix.
		
		Expected behavior is that A is a dataframe with neuroids as columns and data points as rows.
		Expected behavior is that B is a numpy array with neuroids as columns and data points as rows.

		Return the p and r values for each neuroid in a tuple.
		"""

		num_neuroids = self._check_neuroids(A, B)

		metric_over_neuroids = []
		for neuroid in range(num_neuroids):

			# Make sure we index correctly depending on whether it is a pandas dataframe or a numpy array
			A_neuroid = self._index_into_df_or_ndarray(A, neuroid)
			B_neuroid = self._index_into_df_or_ndarray(B, neuroid)

			if self.metric_name == 'pearsonr':
				metric_over_neuroids.append(pearsonr(A_neuroid, B_neuroid))
			elif self.metric_name == 'spearmanr':
				metric_over_neuroids.append(spearmanr(A_neuroid, B_neuroid))
			elif self.metric_name == 'kendalltau':
				metric_over_neuroids.append(kendalltau(A_neuroid, B_neuroid))
			else:
				raise ValueError(
					f'Metric {self.metric_name} not implemented. Please choose from: pearsonr, spearmanr, kendalltau')

		return metric_over_neuroids

	def _get_similarity_matrix(self, A: typing.Union[pd.DataFrame, np.ndarray] = None):
		"""Compute similarity matrix of A. Assumes A [stim; neuroids] and computes a similarity matrix [stim; stim]"""

		A_corr = pd.DataFrame(data=A.T).corr(method=self.rsm_metric)

		return A_corr

	def _get_upper_triangular_indices(self, A):
		"""Given a square matrix A, get the upper triangular indices"""

		# Check matrix sizes
		n_row = A.shape[0]
		idx_upper = np.triu_indices(n_row, 1)  # idx of upper triangular part, above diagonal

		# Check that the indices are correct
		assert (len(idx_upper[0]) == n_row * (n_row - 1) / 2)

		return idx_upper

	def _get_upper_triangular_matrices(self, A, B):

		assert (A.shape == B.shape)

		upper_indices = self._get_upper_triangular_indices(A)

		A_upper = np.asarray(A)[upper_indices]
		B_upper = np.asarray(B)[upper_indices]

		if np.sum(np.isnan(A_upper)) + np.sum(np.isnan(B_upper)) > 0:
			nan_mask = np.logical_or(np.isnan(A_upper), np.isnan(B_upper))
			print(f'Nan values in upper triangular matrix: {np.sum(nan_mask)}')

			A_upper = A_upper[~nan_mask]
			B_upper = B_upper[~nan_mask]

		return np.expand_dims(A_upper, 1), np.expand_dims(B_upper, 1)


class BrainEncoder:
	def __init__(self) -> None:
		pass

	def encode(self,
			   stimset: pd.DataFrame = None,
			   stim_col: str = 'sentence',
			   neural_data: pd.DataFrame = None,
			   specific_target: str = None, ):
		self.specific_target = specific_target
		if specific_target:
			neural_data = pd.DataFrame(neural_data[specific_target], columns=[specific_target])

		self.stimset = stimset
		self.stim_col = stim_col
		self.encoded_brain = neural_data

		return self.encoded_brain


class Preprocessor:
	def __init__(self, preprocess: typing.Union[str, bool, None] = None,
				 **kwargs) -> None:
		from sklearn.pipeline import Pipeline

		preprocessor_classes = {
			'demean': StandardScaler(with_std=False),
			'demean_std': StandardScaler(with_std=True),
			'minmax': MinMaxScaler,
			# Create pipeline for pca
			'pca10': Pipeline([('scaler', StandardScaler(with_std=False)), ('pca', PCA(n_components=10))]),
			'pca800': Pipeline([('scaler', StandardScaler(with_std=False)), ('pca', PCA(n_components=800))]),
			None: None
		}

		if preprocess not in preprocessor_classes:
			raise ValueError(f'Preprocess setting {preprocess} does not exist in preprocessor_classes')

		self.unfitted_scaler = preprocessor_classes[preprocess]
		self.preprocess_name = preprocess

	def fit(self, A_raw: typing.Union[pd.DataFrame, np.ndarray] = None):
		"""Fit based on the input data (A_raw), return scaler. Do not transform.
		
		If the scaler does not exist, return None
		"""
		if self.unfitted_scaler is not None:
			print(f'\nFitting scaler {self.unfitted_scaler}')
			fitted_scaler = self.unfitted_scaler.fit(A_raw)  # demeans column-wise (i.e. per neuroid)
		else:
			fitted_scaler = None

		return fitted_scaler

	def transform(self, scaler: typing.Union[StandardScaler, MinMaxScaler] = None,
				  A_raw: typing.Union[pd.DataFrame, np.ndarray] = None):
		"""Input an array/dataframe (A_raw) and scale based on the transform fitted supplied in scaler.
		If a dateframe is input, then add indexing back after scaling
		
		If scaler is None, then return A_raw.
		
		"""

		if scaler is not None:
			print(f'\nTransforming on new data using scaler {scaler}')
			A_scaled = scaler.transform(A_raw)

			if type(A_raw) == pd.DataFrame:
				if self.preprocess_name.startswith(
						'pca'):  # If PCA, we can't add back the column names because there are now fewer columns
					A_scaled = pd.DataFrame(data=A_scaled, index=A_raw.index)
				else:
					A_scaled = pd.DataFrame(A_scaled, index=A_raw.index, columns=A_raw.columns)

		else:
			print(f'Scaler is None, return A_raw')
			A_scaled = A_raw

		return A_scaled


class Mapping:
	def __init__(self,
				 ANNEncoder: ANNEncoder = None,
				 ann_layer: typing.Union[int, str] = 11,  # Allow to pass a string if we encode a brain as ANN
				 BrainEncoder: BrainEncoder = None,
				 mapping_class: typing.Union[str, typing.Any] = None,
				 metric: Metric = None,
				 Preprocessor: Preprocessor = None,
				 preprocess_X: bool = False,
				 preprocess_y: bool = False,
				 ) -> None:

		self.ann_encoder = ANNEncoder
		self.ann_layer = ann_layer
		self.brain_encoder = BrainEncoder
		self.metric = metric
		self.preprocessor = Preprocessor
		self.preprocess_X = preprocess_X
		self.preprocess_y = preprocess_y

		### Checks ###
		self._check_stimset()
		self._check_ANN_encoder()

		mapping_classes = {
			'ridge': (Ridge, {'alpha': 1.0}),
			'ridgeCV': (RidgeCV, {'alphas': [10 ** x for x in range(-30, 30)], 'alpha_per_target': True}),
			'linear': (LinearRegression, {}),
			'rsa': 'rsa',
			# We do not *fit* an RSA model, we just compute the correlation. We retain the same interface as other models
			None: None}
		self.mapping_class_name = mapping_class
		self.mapping_class = mapping_classes[mapping_class]
		if not self.mapping_class:
			raise ValueError(f'Mapping class not specified')

	def _check_stimset(self):
		if self.ann_encoder.stimset is None:
			raise ValueError('ANN stimset not specified')
		if self.brain_encoder.stimset is None:
			raise ValueError('Brain stimset not specified')

		assert (self.ann_encoder.stimset.index == self.brain_encoder.stimset.index).all()
		assert (self.ann_encoder.stimset[self.ann_encoder.stim_col] == self.brain_encoder.stimset[
			self.brain_encoder.stim_col]).all()

		if self.ann_encoder.stim_col != self.brain_encoder.stim_col:
			print(f'Stimset columns do not match: {self.ann_encoder.stim_col} != {self.brain_encoder.stim_col}')

		print(f'== PASSED stimset checks')

	def _check_ANN_encoder(self):
		assert (self.ann_encoder.encoded_ann is not None)
		assert (self.ann_encoder.encoded_ann[self.ann_layer] is not None)

		# If ann_layer is a string (i.e., an ROI), then we need to make sure we do not end up with a Series object:
		if type(self.ann_layer) == str:
			X = self.ann_encoder.encoded_ann[[self.ann_layer]]
		else:
			X = self.ann_encoder.encoded_ann[self.ann_layer]

		print(f'== PASSED ANN encoder checks')
		print(f'\nANN layer {self.ann_layer} '
			  f'has {X.shape[1]} units'
			  f' for {X.shape[0]} stimuli samples.')

	def _check_neuroids(self,
						A: pd.DataFrame = None,
						B: pd.DataFrame = None):
		"""Check whether columns of A (e.g., y_train) and B (e.g., y_test) are the same.
		Check whether these columns (neuroids) match the brain_encoder.encoded_brain.columns."""

		# Assert that all columns (neuroids) match up
		assert (A.columns == B.columns).all()
		assert (self.brain_encoder.encoded_brain.columns == A.columns).all()

		print(f'== PASSED neuroid checks')

	def _get_column_index(self,
						  A: pd.DataFrame = None,
						  target_col: str = None, ):
		"""Given a pandas df, returns the column index of the target column."""
		col_index = np.argmax(A.columns == target_col)
		col_index2 = A.columns.get_loc(target_col)

		assert (col_index == col_index2)

		return col_index

	def _plot_pred_vs_actual(self,
							 actual: typing.Any = None,
							 pred: typing.Any = None,
							 plot_target: str = None,
							 score: typing.Union[list, np.ndarray] = None, ) -> None:
		"""Plot predicted vs actual values"""

		# Obtain the plot target of interest
		plot_target_index = self._get_column_index(A=actual,
												   target_col=plot_target)
		actual = actual[plot_target].values

		# Find column index and index in the same way in pred
		if pred.shape[1] == 1:
			pred = pred
		else:
			pred = pred[:, plot_target_index]

		# Find the score of interest
		score = score[plot_target_index]

		fig, ax = plt.subplots(figsize=(7, 5))
		ax.set_box_aspect(1)
		plt.scatter(pred, actual, s=20, alpha=0.9)
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		if score:
			plt.annotate(f'{self.metric.metric_name}: {score:.2f}', xy=(0.9, 0.9), xytext=(0.8, 0.04),
						 xycoords='axes fraction', textcoords='axes fraction',
						 horizontalalignment='center', verticalalignment='center')
		plt.title(f'Predicted vs Actual ({len(pred)} data points)')
		plt.show()

	def permute_X(self,
				  X: pd.DataFrame = None,
				  method: str = 'shuffle_X_rows',
				  random_state: int = 0,
				  ) -> pd.DataFrame:
		"""Permute the features of X.
		
		Parameters
		----------
		X : pd.DataFrame
			The dataframe to permute [sentences; features]
		method : str
			The method to use for permutation.
			'shuffle_X_rows' : Shuffle the rows of X (=shuffle the sentences and create a mismatch between the sentence embeddings and target)
			'shuffle_X_cols' : Shuffle the columns of X (=shuffle the neuroids in the same way for each sentence)
			'shuffle_each_X_col': For each column (=feature/unit) of X, permute that feature's values across all sentences.
								  Retains the statistics of the original features (e.g., mean per feature) but the values of the features are shuffled for each sentence.
		random_state : int
			The seed for the random number generator.
		
		Returns
		-------
		pd.DataFrame
			The permuted dataframe.
		"""
		X_orig = X.copy(deep=True)

		print(f' !!!!! Permuting X using: {method} !!!!!')

		# Shuffle the rows of X (=shuffle the sentences and create a mismatch between the sentence embeddings and target)
		if method == 'shuffle_X_rows':
			X = X.sample(frac=1, random_state=random_state)

		# Shuffle the columns of X (=shuffle the neuroids and destroy the sentence embeddings in the same way for each sentence)
		elif method == 'shuffle_X_cols':
			X_ndarray = X.values
			np.random.seed(random_state)
			# Permute columns of X
			X_ndarray = X_ndarray[:, np.random.permutation(X_ndarray.shape[1])]
			X = pd.DataFrame(X_ndarray, columns=X.columns, index=X.index)

		# plt.plot(X_orig.mean(axis=0))
		# plt.plot(pd.DataFrame(X_ndarray).mean(axis=0))
		# plt.show()

		elif method == 'shuffle_each_X_col':
			np.random.seed(random_state)
			for col in X.columns:
				X[col] = np.random.permutation(X[col])

		# plt.plot(X_orig.mean(axis=0))
		# plt.plot(X.mean(axis=0))
		# plt.show()

		else:
			raise ValueError(f'Invalid method: {method}')

		return X

	def CV_score(self,
				 random_state: int = 0,
				 k: int = 5,
				 plot: typing.Union[str, bool] = 'lang_LH_netw',
				 store_pred_per_fold: bool = False,
				 permute_X: typing.Union[str, None] = None):
		"""Run K-fold CV.

		Args
			random_state (int): The seed for the random number seed in KFold.
			k (int): The number of CV folds.
			plot (str, False): Whether to plot the predicted vs actual values for the str value supplied (if not False).
			store_pred_per_fold (bool): Whether to store the predictions per fold.
			permute_X (str, None): Whether to permute the features of X. See self.permute_X() for more info.

		Returns
			df_scores (pd.DataFrame): The summary df. Contains CV_score_{mean,median,std/sem} for the score obtained across folds.
				Rows are neuroids, columns are the scores and metadata.
			df_scores_across_folds (pd.DataFrame): The df containing the scores obtained per fold.
				Rows are fold x neuroid (multindexed), columns are the scores and metadata.
			d_CV_pred (dict): Contains:
				key = y with value = pd.DataFrame of the y (neural values) that went into the CV
				(rows are items, columns are neuroids).
				key = y_pred with value = pd.DataFrame of the y_pred (predicted neural values) obtained from the CV

		"""

		# Classifier
		clf = self.mapping_class[0](**self.mapping_class[1])

		# Regressors (X) and targets (y)
		# If ann_layer is a string (i.e., an ROI), then we need to make sure we do not end up with a Series object:
		if type(self.ann_layer) == str:
			X = self.ann_encoder.encoded_ann[[self.ann_layer]]
		else:
			X = self.ann_encoder.encoded_ann[self.ann_layer]
		y = self.brain_encoder.encoded_brain

		# Checks: perturbing the regressors (X)
		if permute_X is not None:
			X = self.permute_X(X=X,
							   method=permute_X,
							   random_state=random_state)

		# Train/test indices
		kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

		train_indices = []
		test_indices = []

		scores_across_folds = []  # storing the score between y_test and y_pred in each fold
		scores_across_folds_raw = []  # storing the score between y_test and y_pred in each fold (without changing nan to 0)
		p_across_folds = []  # storing the p-value associated with the score between y_test and y_pred in each fold
		p_across_folds_raw = []  # storing the p-value associated with the score between y_test and y_pred in each fold (without changing nan to 0)
		alpha_across_folds = []  # storing the alpha value identified in the test split in each fold

		y_tests = []  # storing the y_test values in each fold (for asserting that they match up with y in the end)
		y_preds_cv = []  # storing the y_pred values in each fold (for storing them in a dict structure with keys "y"
		# and "y_pred-CV-k-{k}" for each fold)

		d_cv_log = defaultdict()

		for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
			test_indices.append(test_index)
			train_indices.append(train_index)

			X_train, X_test = X.iloc[train_index], X.iloc[test_index]
			y_train, y_test = y.iloc[train_index], y.iloc[test_index]

			# Preprocessing
			if self.preprocess_X:
				X_scaler = self.preprocessor.fit(X_train)  # Fit transform on train set to avoid data leakage
				X_train = self.preprocessor.transform(scaler=X_scaler, A_raw=X_train)
				X_test = self.preprocessor.transform(scaler=X_scaler,
													 A_raw=X_test)  # use transform from training set on the test set
			if self.preprocess_y:
				y_scaler = self.preprocessor.fit(y_train)  # Fit transform on train set to avoid data leakage
				y_train = self.preprocessor.transform(scaler=y_scaler, A_raw=y_train)
				y_test = self.preprocessor.transform(scaler=y_scaler,
													 A_raw=y_test)  # use transform from training set on the test set

			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			y_pred = pd.DataFrame(y_pred, index=y_test.index,  # Package into df for easier handling
								  columns=y_test.columns)

			fold_score_and_p = self.metric._metric_over_neuroids(A=y_test,
																 B=y_pred)
			fold_score_raw = [x[0] for x in fold_score_and_p]
			fold_p_raw = [x[1] for x in fold_score_and_p]

			# Convert nans to 0 (and check that it occurs due to a constant y or y_pred array)
			fold_score, fold_p = check_constant_y_ypred(y_pred=y_pred,
														y=y_test,
														score=fold_score_raw,
														p=fold_p_raw, )

			# Append scores, p-vals, y_test, y_pred to lists
			scores_across_folds.append(fold_score)
			scores_across_folds_raw.append(fold_score_raw)
			p_across_folds.append(fold_p)
			p_across_folds_raw.append(fold_p_raw)
			y_tests.append(y_test)
			y_preds_cv.append(y_pred)

			if plot is not None:
				self._plot_pred_vs_actual(actual=y_test,
										  pred=y_pred,
										  score=fold_score,
										  plot_target=plot)

			# Logging: Create a log of what happened per fold
			self._check_neuroids(A=y_train,
								 B=y_test)

			assert (y_train.columns.equals(y_test.columns))
			neuroid_col_indexer = y_train.columns.values  # Indexer for neuroid columns. Contains the neuroid names

			if permute_X is None:  # only assert if we do not permute the X
				assert (y_train.index == X_train.index).all()
				assert (y_test.index == X_test.index).all()
			sampleid_train_row_indexer = y_train.index.values  # Indexer for sampleid rows. Contains the sample identifiers
			sampleid_test_row_indexer = y_test.index.values  # Indexer for sampleid rows. Contains the sample identifiers

			df_fold_log = pd.DataFrame({'CV_fold_idx': fold_idx,
										'CV_fold_score': fold_score,
										'CV_fold_score_raw': fold_score_raw,
										'CV_fold_p': fold_p,
										'CV_fold_p_raw': fold_p_raw,

										'train_indices': [train_index],
										'train_indices_sampleid': [sampleid_train_row_indexer],
										# Which sampleids/itemids were used for train
										'test_indices': [test_index],
										'test_indices_sampleid': [sampleid_test_row_indexer],
										# Which sampleids/itemids were used for test

										'train_actual_mean': np.mean(y_train, axis=0).values,
										'test_actual_mean': np.mean(y_test, axis=0).values,
										'test_pred_mean': np.mean(y_pred, axis=0).values, },

									   index=neuroid_col_indexer)

			if self.mapping_class_name.startswith('ridge'):
				df_fold_log['alpha'] = clf.alpha_
				alpha_across_folds.append(clf.alpha_)

			# Append y_test and y_pred responses per fold
			if store_pred_per_fold:
				# Make y_test into a list of lists such that we can store it in the big df_fold_log
				y_test_list_per_fold = [y_test[neuroid].values for neuroid in neuroid_col_indexer]
				y_pred_list_per_fold = [y_pred[neuroid].values for neuroid in neuroid_col_indexer]

				df_fold_log['actual_responses'] = y_test_list_per_fold
				df_fold_log['pred_responses'] = y_pred_list_per_fold

			d_cv_log[fold_idx] = df_fold_log

		print(f'\nFinished {k} CV folds!\n')

		# Convert scores into array
		scores_arr = np.array(scores_across_folds)
		scores_arr_raw = np.array(scores_across_folds_raw)
		p_arr = np.array(p_across_folds)
		p_arr_raw = np.array(p_across_folds_raw)
		df_scores_across_folds = pd.concat(d_cv_log)

		# Quantify how many nans are in each column (fold) of the scores array
		nan_count_per_fold = np.sum(np.isnan(scores_arr_raw), axis=0)

		df_scores = pd.DataFrame({'CV_score_mean': np.mean(scores_arr, axis=0),
								  # Should be no nans here and in the other score_arr computations
								  'CV_score_raw_nanmean': np.nanmean(scores_arr_raw, axis=0),
								  'CV_p_mean': np.mean(p_arr, axis=0),
								  'CV_p_raw_nanmean': np.nanmean(p_arr_raw, axis=0),
								  'CV_score_median': np.median(scores_arr, axis=0),
								  'CV_score_raw_nanmedian': np.nanmedian(scores_arr_raw, axis=0),
								  'CV_p_median': np.nanmedian(p_arr, axis=0),
								  'CV_p_raw_nanmedian': np.nanmedian(p_arr_raw, axis=0),
								  'CV_score_std': np.std(scores_arr, axis=0),
								  # Compute std/sem over the fold scores without nans (with 0s)
								  'CV_score_sem': nansem(scores_arr, axis=0),  # nansem implementes default pandas sem
								  'CV_score_raw_nanstd': np.nanstd(scores_arr_raw, axis=0),
								  'CV_score_raw_nansem': nansem(scores_arr_raw, axis=0),
								  'CV_nan_count_per_fold': nan_count_per_fold, },

								 index=neuroid_col_indexer)

		if self.mapping_class_name.startswith('ridge'):  # Get mean alpha across folds
			alphas_arr = np.array(alpha_across_folds)
			df_scores['CV_alpha_mean'] = np.mean(alphas_arr, axis=0)
			df_scores['CV_alpha_median'] = np.median(alphas_arr, axis=0)

		# Store a simple dict structure with keys "y" and "y_pred-CV-k-{k}" which has the actual value and predicted
		# value for each item, obtained across all folds

		# First make sure that y_tests and y are the same
		df_y_tests = pd.concat(y_tests)  # Concat the list of dfs in y_tests
		df_y_tests = df_y_tests.reindex(y.index)
		assert (df_y_tests.equals(y))

		df_y_preds = pd.concat(y_preds_cv)  # Concat the list of dfs in y_preds_cv
		df_y_preds = df_y_preds.reindex(y.index)

		# Store the y and y_pred-CV-k-{k} in a dict (similar to the y_pred_full dict)
		d_CV_pred = self.package_cv_y_pred(y=df_y_tests,
										   y_pred=df_y_preds,
										   cv_str=f'CV-k-{k}')

		return df_scores, df_scores_across_folds, d_CV_pred

	def fit_full_mapping(self, **kwargs):

		# Classifier
		clf = self.mapping_class[0](**self.mapping_class[1])

		# Regressors (X) and targets (y)
		# If ann_layer is a string (i.e., an ROI), then we need to make sure we do not end up with a Series object:
		if type(self.ann_layer) == str:
			X = self.ann_encoder.encoded_ann[[self.ann_layer]]
		else:
			X = self.ann_encoder.encoded_ann[self.ann_layer]
		y = self.brain_encoder.encoded_brain
		neuroid_order = y.columns.values

		# Preprocessing (using the full set). If no preprocess_X and preprocess_y, return scalers as None, and arrays as raw.
		X_scaler, y_scaler = None, None

		if self.preprocess_X:
			X_scaler = self.preprocessor.fit(X)  # Given that we are fitting the full mapping, just transform it all
			X = self.preprocessor.transform(scaler=X_scaler, A_raw=X)
		if self.preprocess_y:
			y_scaler = self.preprocessor.fit(y)
			y = self.preprocessor.transform(scaler=y_scaler, A_raw=y)

		clf.fit(X, y)

		# Predict using the full model
		y_pred_full = clf.predict(X)
		assert (X.index.values == y.index.values).all()
		assert (neuroid_order == y.columns.values).all()
		y_pred_full = pd.DataFrame(y_pred_full, columns=neuroid_order,
								   index=X.index)  # Package into df for easier handling

		# Package df full scores
		full_score_and_p = self.metric._metric_over_neuroids(A=y,
															 B=y_pred_full)
		full_score_raw = [x[0] for x in full_score_and_p]
		full_p_raw = [x[1] for x in full_score_and_p]

		# Convert nans to 0 (and check that it occurs due to a constant y or y_pred array)
		full_score, full_p = check_constant_y_ypred(y_pred=y_pred_full,
													y=y,
													score=full_score_raw,
													p=full_p_raw, )

		df_full_scores = pd.DataFrame({'full_score': full_score,
									   'full_score_raw': full_score_raw,
									   'full_p': full_p,
									   'full_p_raw': full_p_raw},
									  index=neuroid_order)

		if self.mapping_class_name.startswith('ridge'):  # Get alpha
			df_full_scores['full_alpha'] = clf.alpha_

		# Package df with weights
		d_weights = self.package_full_mapping(clf_fitted=clf,
											  neuroid_order=neuroid_order,
											  X_scaler=X_scaler,
											  y_scaler=y_scaler,
											  **kwargs)

		d_full_y_pred = self.package_full_y_pred(y=y,
												 y_pred=y_pred_full)

		return df_full_scores, d_weights, d_full_y_pred

	def package_full_mapping(self, clf_fitted: typing.Any = None,
							 neuroid_order: np.ndarray = None,
							 X_scaler: typing.Union[StandardScaler, MinMaxScaler, None] = None,
							 y_scaler: typing.Union[StandardScaler, MinMaxScaler, None] = None,
							 **kwargs):
		"""Store the mapping classifier (weights, alphas (if any), and intercept)

		Args
			clf_fitted (sklearn classifier): Fitted classifier
			neuroid_order (np.ndarray): Order of the neuroids in the y array
			X_scaler (sklearn scaler): Fitted scaler for X (if any)
			y_scaler (sklearn scaler): Fitted scaler for y (if any)

		Returns
			d (dict): Dictionary with keys "clf", "X_scaler", "y_scaler", "neuroid_order"

		"""

		d = {}

		d['clf'] = clf_fitted  # stores both weights and alphas

		d['X_scaler'] = X_scaler
		d['y_scaler'] = y_scaler

		d['neuroid_order'] = neuroid_order

		return d

	def load_full_mapping(self, WEIGHTDIR,
						  mapping_result_identifier: str = None,
						  mapping_save_str: str = None,
						  **kwargs):
		"""Load the mapping classifier (weights, alphas (if any), and intercept)
		
		Load the neuroid order as well as the preprocessor transform (if any)

		Args
			WEIGHTDIR (str): Directory where the weights (full_mapping) are stored
			mapping_result_identifier (str): Result string identifier (folder) for the weights/mapping
			mapping_save_str (str): String identifier that was used to store the weights/mapping

		"""

		with open(join(WEIGHTDIR, 'fit_mapping', mapping_result_identifier, mapping_save_str), 'rb') as f:
			d = pickle.load(f)

		self.prefitted_clf = d['clf']
		self.prefitted_clf_neuroid_order = d['neuroid_order']
		self.prefitted_X_scaler = d['X_scaler']
		self.prefitted_y_scaler = d['y_scaler']


	def predict_using_prefitted_mapping(self):
		"""Perform predictions on the brain stimset based on a prefitted clf.
		
		Return predictions per item (rows), per neuroid (columns)
		
		"""
		assert (self.prefitted_clf is not None)

		# Use the ann_encoder.encoded_ann to obtain regressors (X) of the new target brain stimset
		X = self.ann_encoder.encoded_ann[self.ann_layer]

		# Pass X through the scaler. If prefitted_X_scaler is None, return X_raw
		X = self.preprocessor.transform(scaler=self.prefitted_X_scaler,
										A_raw=X)

		if self.prefitted_X_scaler is None:
			assert (X.equals(self.ann_encoder.encoded_ann[self.ann_layer]))

		preds = self.prefitted_clf.predict(X)

		assert (preds.shape[0] == self.ann_encoder.stimset.shape[0])
		assert (preds.shape[0] == self.brain_encoder.stimset.shape[0])

		assert (preds.shape[1] == len(
			self.prefitted_clf_neuroid_order))  # make sure that a correct number of neuroids were predicted

		# Package into a df with correct neuroid (col) names
		df_preds = pd.DataFrame(preds, columns=self.prefitted_clf_neuroid_order, index=X.index)

		return df_preds

	def package_full_y_pred(self, y: pd.DataFrame = None,
							y_pred: pd.DataFrame = None):
		"""Store the predicted y. Adds neuroid order to y_pred based on y.

		Args
			y (pd.DataFrame): y (actual neural values) with rows as items and columns as neuroids
			y_pred (pd.DataFrame): Predicted y (neural values) with rows as items and columns as neuroids

		Returns
			d (dict): Dictionary with keys "y", "y_pred-full". "full" refers to the fact that this prediction was not
				obtained using CV.
		"""

		d = {}
		d['y'] = y
		d['y_pred-full'] = y_pred

		return d

	def package_full_y_pred_using_prefitted_mapping(self, y: pd.DataFrame = None,
													y_pred: pd.DataFrame = None):
		"""Package the full y_pred using the prefitted mapping (y_pred is already a df)

		Args
			y (pd.DataFrame): y (actual neural values) with rows as items and columns as neuroids
			y_pred (pd.DataFrame): Predicted y (neural values) with rows as items and columns as neuroids

		Returns
			d (dict): Dictionary with keys "y", "y_pred-full". "full" refers to the fact that this prediction was not
				obtained using CV.
		"""

		assert (y_pred.index.equals(y.index))
		assert (y_pred.columns.equals(y.columns))

		d = {}
		d['y'] = y
		d['y_pred-full'] = y_pred

		return d

	def package_cv_y_pred(self, y: pd.DataFrame = None,
						  y_pred: pd.DataFrame = None,
						  cv_str: str = None):
		"""Package the y_pred obtained from CV (y_pred is already a df)

		Args
			y (pd.DataFrame): y (actual neural values) with rows as items and columns as neuroids
			y_pred (pd.DataFrame): Predicted y (neural values) with rows as items and columns as neuroids
			cv_str (str): String identifier for the CV (e.g. "CV-k-5")

		Returns
			d (dict): Dictionary with keys "y", "y_pred-{cv_str}". "CV" refers to the fact that this prediction was
				obtained using CV.
		"""

		assert (y_pred.index.equals(y.index))
		assert (y_pred.columns.equals(y.columns))

		d = {}
		d['y'] = y
		d[f'y_pred-{cv_str}'] = y_pred

		return d

	def RSA_score(self,
				  permute_X: bool = False,
				  permute_X_seed: int = 0,
				  cache_rsms: bool = True,
				  X_rsm_identifier: str = '',
				  y_rsm_identifier: str = '',
				  RSMDIR: str = '',
				  ):
		"""
		Calculate RSA between the RSMs of the ANN and the brain.

		Args:
			permute_X (bool): Whether to permute the X (ANN) RSM before calculating RSA (for generating null distribution)
			permute_X_seed (int): Random seed for permuting X (ANN) RSM
			cache_RSMs (bool): Whether to cache the RSMs of the ANN and the brain.
		"""

		#### Check if RSMs are cached, otherwise compute ####

		# X RSM
		X_rsm_path = os.path.join(RSMDIR, f'{X_rsm_identifier}.pkl')
		if os.path.exists(X_rsm_path):
			print(f'Loading cached X RSM from {X_rsm_path}')
			with open(X_rsm_path, 'rb') as f:
				X_RSM = pickle.load(f)

		else:
			X = self.ann_encoder.encoded_ann[self.ann_layer]

			# Preprocess
			if self.preprocess_X:
				X_scaler = self.preprocessor.fit(X) # Fit and transform on X
				X = self.preprocessor.transform(scaler=X_scaler, A_raw=X)

			# Permute
			if permute_X:
				X = X.sample(frac=1, random_state=permute_X_seed).reset_index(drop=True, inplace=False)
				print(f'Permuting rows of X before calculating RSA using random seed {permute_X_seed}')

			X_RSM = self.metric._get_similarity_matrix(X)

			if cache_rsms:
				if not os.path.exists(RSMDIR):
					os.makedirs(RSMDIR)

				with open(X_rsm_path, 'wb') as f:
					pickle.dump(X_RSM, f)
				print(f'Cached X RSM at {X_rsm_path}')

		# y RSM
		y_rsm_path = os.path.join(RSMDIR, f'{y_rsm_identifier}.pkl')
		if os.path.exists(y_rsm_path):
			print(f'Loading cached y RSM from {y_rsm_path}')
			with open(y_rsm_path, 'rb') as f:
				y_RSM = pickle.load(f)

		else:
			y = self.brain_encoder.encoded_brain

			# Preprocess
			if self.preprocess_y:
				y_scaler = self.preprocessor.fit(y)
				y = self.preprocessor.transform(scaler=y_scaler, A_raw=y)

			y_RSM = self.metric._get_similarity_matrix(y)

			if cache_rsms:
				with open(y_rsm_path, 'wb') as f:
					pickle.dump(y_RSM, f)
				print(f'Cached y RSM at {y_rsm_path}')

		# Simple visual check
		# plt.imshow(X_RSM, interpolation='none')
		# plt.show()

		#### Obtain upper triangular indices ####
		X_RSM_upper, y_RSM_upper = self.metric._get_upper_triangular_matrices(A=X_RSM, B=y_RSM)

		#### Then, use metric to calculate RSA ####
		score = self.metric._metric_over_neuroids(A=X_RSM_upper, B=y_RSM_upper)  # Return as list
		assert (len(score) == 1)  # We only expect one score per comparison
		score = score[0]  # Unpack from list

		# Package into a df with the score and p-value
		df_score = pd.DataFrame({'score': score[0],
								 'p': score[1]},
								index=[0])

		return df_score


class ArgumentLogger:
	"""
	Class for logging argument from argparse. Takes the args as a dictionary.
	
	Generate result_identifiers and save strings based on which particular script is being run.
	
	"""

	def __init__(self, args: dict,
				 script_name: typing.Union[str],
				 result_root: typing.Union[str],
				 weight_root: typing.Union[str, None] = None,
				 log_root: typing.Union[str, None] = None,
				 actv_root: typing.Union[str, None] = None,
				 add_args: dict = None,
				 ):

		self.args = args
		self.script_name = script_name

		self.create_result_identifier()

		self.RESULTDIR = join(result_root, self.script_name, self.result_identifier)
		self.WEIGHTDIR = join(weight_root, self.script_name, self.result_identifier)
		self.LOGDIR = join(log_root, self.script_name, self.result_identifier)
		if actv_root is not None:
			self.ACTVDIR = join(actv_root, self.args["source_model"], self.args["sent_embed"])
		else:
			self.ACTVDIR = None
		self.make_dirs()

		self.d_DIRS = {'RESULTDIR': self.RESULTDIR,
					   'WEIGHTDIR': self.WEIGHTDIR,
					   'LOGDIR': self.LOGDIR,
					   'ACTVDIR': self.ACTVDIR}

		# Add additional arguments if supplied
		self.add_key_val(add_args)

	def make_dirs(self):
		"""Create three main directories for storing / logging:
		"""

		os.makedirs(self.RESULTDIR, exist_ok=True)
		os.makedirs(self.LOGDIR, exist_ok=True)

		if 'fit_mapping' in self.script_name:
			os.makedirs(self.WEIGHTDIR, exist_ok=True)

	def add_args_to_df(self, df: pd.DataFrame):
		"""Given an input df, add arguments as columns"""

		for k in self.args.keys():
			df[k] = self.args[k]

		return df

	def add_date(self, df: pd.DataFrame):
		"""Write the date to a column in df"""

		df['date'] = datetime.now().strftime("%Y-%m-%d_%H:%M")

		return df

	def add_key_val(self, d: dict = None):
		"""Add key-value pair to the argument dictionary"""

		if d is not None:
			for k, v in d.items():
				self.args[k] = v
			# print(f'Added key: {k} to args dict')

	def create_result_identifier(self):
		"""Generate a result identifier (using to create result folder to store outputs)
		Add the result identifier to the namespace argument dictionary
		"""

		if self.script_name == 'fit_mapping' or self.script_name == 'fit_mapping_pretransformer':
			self.result_identifier = f'SOURCE-{self.args["source_model"]}_' \
									 f'{self.args["sent_embed"]}_' \
									 f'TARGET-{self.args["regression_dict_type"]}-' \
									 f'{self.args["UID"]}_' \
									 f'{self.args["sess_num"]}_' \
									 f'{self.args["FL"]}_' \
									 f'{self.args["func_thresh"]}_' \
									 f'MAPPING-{self.args["mapping_class"]}-' \
									 f'{self.args["metric"]}'

		elif self.script_name == 'use_mapping':
			self.result_identifier = f'SOURCE-{self.args["source_model"]}_' \
									 f'{self.args["sent_embed"]}_' \
									 f'TARGET-{self.args["mapping_regression_dict_type"]}-' \
									 f'{self.args["mapping_UID"]}_' \
									 f'{self.args["mapping_sess_num"]}_' \
									 f'{self.args["mapping_FL"]}_' \
									 f'{self.args["mapping_func_thresh"]}_' \
									 f'MAPPING-{self.args["mapping_class"]}-' \
									 f'{self.args["metric"]}'

		elif self.script_name == 'use_mapping_search':
			self.result_identifier = f'SOURCE-{self.args["source_model"]}_' \
									 f'{self.args["sent_embed"]}_' \
									 f'TARGET-{self.args["mapping_regression_dict_type"]}-' \
									 f'{self.args["mapping_UID"]}_' \
									 f'{self.args["mapping_sess_num"]}_' \
									 f'{self.args["mapping_FL"]}_' \
									 f'{self.args["mapping_func_thresh"]}_' \
									 f'MAPPING-{self.args["mapping_class"]}-' \
									 f'{self.args["metric"]}'

		elif self.script_name == 'use_mapping_external' or self.script_name == 'use_mapping_external_pretransformer':
			self.result_identifier = f'SOURCE-{self.args["source_model"]}_' \
									 f'{self.args["sent_embed"]}_' \
									 f'TARGET-{self.args["mapping_regression_dict_type"]}-' \
									 f'{self.args["mapping_UID"]}_' \
									 f'{self.args["mapping_sess_num"]}_' \
									 f'{self.args["mapping_FL"]}_' \
									 f'{self.args["mapping_func_thresh"]}_' \
									 f'MAPPING-{self.args["mapping_class"]}-' \
									 f'{self.args["metric"]}'

		elif self.script_name == 'run_rsa':
			self.result_identifier = f'SOURCE-{self.args["source_model"]}_' \
									 f'{self.args["sent_embed"]}_' \
									 f'TARGET-{self.args["regression_dict_type"]}-' \
									 f'{self.args["UID"]}_' \
									 f'{self.args["sess_num"]}_' \
									 f'{self.args["FL"]}_' \
									 f'{self.args["func_thresh"]}_' \
									 f'MAPPING-{self.args["mapping_class"]}-' \
									 f'{self.args["metric"]}-' \
									 f'{self.args["rsm_metric"]}'

		else:
			raise ValueError(f'Analysis mode not recognized')

		self.add_key_val({'result_identifier': self.result_identifier})

	def create_save_str(self):
		"""Generate a string to be used for saving outputs (e.g., name for pickle files / plot names).
		Add the save_str to the namespace argument dictionary"""

		if self.script_name == 'fit_mapping' or self.script_name == 'fit_mapping_pretransformer':
			self.save_str = f'SOURCE-{self.args["source_layer"]}_' \
							f'TARGET-{self.args["savestr_prefix"]}-' \
							f'{self.args["specific_target"]}_' \
							f'{self.args["modeltype"]}-' \
							f'{self.args["preproc"]}-' \
							f'{self.args["pcstop"]}-' \
							f'{self.args["fracs"]}-' \
							f'{self.args["norm"]}_' \
							f'MAPPING-{self.args["preprocessor"]}-' \
							f'{self.args["preprocess_X"]}-' \
							f'{self.args["preprocess_y"]}'

		elif self.script_name == 'use_mapping':
			# First part is the mapping_id (folder), next part is the save str that is target specific
			self.save_str = f'SOURCE-{self.args["source_layer"]}_' \
							f'TARGET-{self.args["mapping_savestr_prefix"]}-' \
							f'{self.args["mapping_specific_target"]}_' \
							f'{self.args["mapping_modeltype"]}-' \
							f'{self.args["mapping_preproc"]}-' \
							f'{self.args["mapping_pcstop"]}-' \
							f'{self.args["mapping_fracs"]}-' \
							f'{self.args["mapping_norm"]}_' \
							f'MAPPING-{self.args["preprocessor"]}-' \
							f'{self.args["preprocess_X"]}-' \
							f'{self.args["preprocess_y"]}/' \
							f'TARGET-{self.args["target_regression_dict_type"]}-' \
							f'{self.args["target_UID"]}-' \
							f'{self.args["target_sess_num"]}-' \
							f'{self.args["target_FL"]}-' \
							f'{self.args["target_func_thresh"]}-' \
							f'{self.args["target_savestr_prefix"]}-' \
							f'{self.args["target_specific_target"]}_' \
							f'{self.args["target_modeltype"]}-' \
							f'{self.args["target_preproc"]}-' \
							f'{self.args["target_pcstop"]}-' \
							f'{self.args["target_fracs"]}-' \
							f'{self.args["target_norm"]}'

		elif self.script_name == 'use_mapping_search':
			# First part is the mapping_id (folder), next part is the save str that is corpus specific
			self.save_str = f'SOURCE-{self.args["source_layer"]}_' \
							f'TARGET-{self.args["mapping_savestr_prefix"]}-' \
							f'{self.args["mapping_specific_target"]}_' \
							f'{self.args["mapping_modeltype"]}-' \
							f'{self.args["mapping_preproc"]}-' \
							f'{self.args["mapping_pcstop"]}-' \
							f'{self.args["mapping_fracs"]}-' \
							f'{self.args["mapping_norm"]}_' \
							f'MAPPING-{self.args["preprocessor"]}-' \
							f'{self.args["preprocess_X"]}-' \
							f'{self.args["preprocess_y"]}/' \
							f'TARGET-{self.args["corpus_filename"]}'

		elif self.script_name == 'use_mapping_external' or self.script_name == 'use_mapping_external_search' or self.script_name == 'use_mapping_external_pretransformer':
			# First part is the mapping_id (folder), next part is the save str that is stimset specific
			stimset_filename = self.args["stimset_filename"].split('.')[0]
			print(f'Dropping file extension from stimset filename, saving using: {stimset_filename}')

			self.save_str = f'SOURCE-{self.args["source_layer"]}_' \
							f'TARGET-{self.args["mapping_savestr_prefix"]}-' \
							f'{self.args["mapping_specific_target"]}_' \
							f'{self.args["mapping_modeltype"]}-' \
							f'{self.args["mapping_preproc"]}-' \
							f'{self.args["mapping_pcstop"]}-' \
							f'{self.args["mapping_fracs"]}-' \
							f'{self.args["mapping_norm"]}_' \
							f'MAPPING-{self.args["preprocessor"]}-' \
							f'{self.args["preprocess_X"]}-' \
							f'{self.args["preprocess_y"]}/' \
							f'TARGET-{stimset_filename}'

		elif self.script_name == 'run_rsa':
			self.save_str = f'SOURCE-{self.args["source_layer"]}_' \
							f'TARGET-{self.args["savestr_prefix"]}-' \
							f'{self.args["specific_target"]}_' \
							f'{self.args["modeltype"]}-' \
							f'{self.args["preproc"]}-' \
							f'{self.args["pcstop"]}-' \
							f'{self.args["fracs"]}-' \
							f'{self.args["norm"]}_' \
							f'MAPPING-{self.args["preprocessor"]}-' \
							f'{self.args["preprocess_X"]}-' \
							f'{self.args["preprocess_y"]}_' \
							f'permuteX-{self.args["permute_X"]}-' \
							f'{self.args["permute_X_seed"]}'

		else:
			print(f'Analysis mode not recognized!')
			self.save_str = 'DEFAULT_SAVE_STR'

		# *IF* We run any fit_mapping or use_mapping scripts with permute_X not None, we need to add that to the save_str
		if self.args["permute_X"] is not None:
			if self.script_name.startswith('fit_mapping') or self.script_name.startswith('use_mapping'):
				self.save_str += f'_permuteX-{self.args["permute_X"]}'

		self.add_key_val({'save_str': self.save_str})

	def create_rsm_identifiers(self, ):
		"""
		Generate RSM identifier for either source or target RSMs.

		(It doesn't matter which one is target or source for RSA, but we just denote them as target and source for clarity)

		"""
		# X
		source_rsm_identifier = f'RSM-{self.args["rsm_metric"]}_' \
								f'{self.args["permute_X"]}-' \
								f'{self.args["permute_X_seed"]}-' \
								f'{self.args["source_model"]}-' \
								f'{self.args["sent_embed"]}-' \
								f'{self.args["source_layer"]}-' \
								f'{self.args["preprocessor"]}-' \
								f'{self.args["preprocess_X"]}'

		# y
		target_rsm_identifier = f'RSM-{self.args["rsm_metric"]}_' \
								f'{self.args["regression_dict_type"]}-' \
								f'{self.args["UID"]}-' \
								f'{self.args["sess_num"]}-' \
								f'{self.args["FL"]}-' \
								f'{self.args["func_thresh"]}-' \
								f'{self.args["savestr_prefix"]}-' \
								f'{self.args["specific_target"]}-' \
								f'{self.args["modeltype"]}-' \
								f'{self.args["preproc"]}-' \
								f'{self.args["pcstop"]}-' \
								f'{self.args["fracs"]}-' \
								f'{self.args["norm"]}-' \
								f'{self.args["preprocessor"]}-' \
								f'{self.args["preprocess_y"]}'  # "norm" is how the neural data were normalized
								# "preprocess_y" is if any posthoc preprocessing is taking place (in the post-packaging stage)

		self.add_key_val({'source_rsm_identifier': source_rsm_identifier,
						 'target_rsm_identifier': target_rsm_identifier})

		self.source_rsm_identifier = source_rsm_identifier
		self.target_rsm_identifier = target_rsm_identifier



	def print_package_versions(self):
		from importlib.metadata import version
		print(f'Transformers: {version("transformers")}\n'
			  f'Pandas: {version("pandas")}\n'
			  f'Numpy: {version("numpy")}\n'
			  f'Scikit: {version("scikit-learn")}\n')

	def create_result_folder(self, folder_path: str = '',
							 **kwargs):
		"""Create result folder by merging the path to the folder (typically RESULTDIR / RESULTIDENTIFIER)
		and the save_str. If save_str contains subfolders, make sure that those are created too.
		
		"""

		# Check if path contains one more subfolder level. If so, create that directory
		if '/' in self.save_str:
			all_subdir_indicators = [m.start() for m in re.finditer('/', self.save_str)]
			assert (len(all_subdir_indicators) == 1)

			save_str_subdir = self.save_str.split("/")[0]
			save_str_str = self.save_str.split("/")[1]  # the actual name of the data we want to save
			os.makedirs(join(folder_path, f'{save_str_subdir}'), exist_ok=True)
			folder_path_save_str = join(folder_path,
										save_str_subdir,
										f'{kwargs.get("prefix_str")}_{save_str_str}'
										f'{kwargs.get("suffix_str")}.pkl')

		else:
			os.makedirs(folder_path, exist_ok=True)

			folder_path_save_str = join(folder_path,
										f'{kwargs.get("prefix_str")}_{self.save_str}'
										f'{kwargs.get("suffix_str")}.pkl')

		return folder_path_save_str

	def dump_data(self, data: typing.Union[pd.DataFrame, dict] = None,
				  path: str = ''):
		"""Dump data. Use pickle protocol 4."""
		if type(data) == pd.DataFrame:
			data.to_pickle(path, protocol=4)
		elif type(data) == dict:
			with open(path, 'wb') as f:
				pickle.dump(data, f, protocol=4)
		else:
			raise ValueError(f'Invalid data type {type(data)}, cannot save!')

		print(f'\nData saved to:\n{path}')

	def store(self, data: typing.Union[pd.DataFrame, dict] = None,
			  DIR: str = 'RESULTDIR',
			  prefix_str: str = '',
			  suffix_str: str = ''):
		"""Store df or dict to directory of interest in the result_identifier subfolder
		
		DIR (str): If the string is already initialized (e.g., "RESULTDIR") then store under that directory.
				   If not recognized, use the supplied string as the root directory.
		"""

		if suffix_str != '' and not suffix_str.startswith('_'):
			suffix_str = '_' + suffix_str

		if DIR in self.d_DIRS:  # If the DIR string exist in the default directories:
			folder_path = join(self.d_DIRS[DIR])
		else:  # Use a manual directory and merge with result_identifier
			folder_path = join(DIR,
							   self.result_identifier)

		# Create the result folder taking subfolders from save_str into account
		folder_path_save_str = self.create_result_folder(folder_path=folder_path,
														 **{'prefix_str': prefix_str,
															'suffix_str': suffix_str})

		self.dump_data(data=data,
					   path=folder_path_save_str)
