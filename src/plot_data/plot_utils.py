import pandas as pd
import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import typing
from textwrap import wrap
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib
# Import resources from the folder up one level
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resources import *

# Set random seed for plot jitters and noise simulation
np.random.seed(0)

def groupby_coord(df: pd.DataFrame,
                  coord_col: str = 'item_id',
                  aggfunc: str = 'mean',
                  ) -> pd.DataFrame:
    """
	Group a pandas dataframe by the coordinates specified in coord_cols.
	Most common use case: group by item_id (across several UIDs).

	This function by default groups the numeric columns according to the aggfunc (mean by default).
	For the string columns, the first value is kept. Importantly, we check whether all string columns are the same for
	each item_id (or other coordinate). Then, we only keep the coords that are shared across all item_ids
	(such that we avoid the scenario where we think we can use a string column as a coordinate, but in reality it was
	different for each occurence of that item_id).

	Args:
		df (pd.DataFrame): dataframe to group
		coord_col (str): column to group by (currently only supports one coord col)
		aggfunc: aggregation function for numeric columns. If std/sem, always use n-1 for ddof

	Returns:
		df_grouped (pd.DataFrame): grouped dataframe
	"""

    df_to_return = df.copy(deep=True)

    # Create df that is grouped by col coord using aggfunc
    # Keep string columns as is (first)
    if aggfunc == 'mean':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    elif aggfunc == 'median':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    elif aggfunc == 'std':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.std(ddof=1) if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    elif aggfunc == 'sem':
        df_grouped = df_to_return.groupby(coord_col).agg(
            lambda x: x.sem(ddof=1) if np.issubdtype(x.dtype, np.number) else x.iloc[0])
    else:
        raise ValueError(f'aggfunc {aggfunc} not supported')

    # Check whether the string arguments were all the same (to ensure correct grouping and metadata for str cols) ###
    shared_str_cols = []
    not_shared_str_cols = []
    for coord_val in df_grouped.index:
        # Get the values of the coordinate of interest in df_to_return
        df_test = df_to_return.loc[df_to_return[coord_col] == coord_val, :]
        df_test_object = df_test.loc[:, df_test.dtypes == object]  # Get the string cols

        # If not nan, check that each col has unique values
        if not df_test_object.isna().all().all():
            for col in df_test_object.columns:
                if len(df_test_object[col].unique()) > 1:
                    # print(f'Column {col} has multiple values for item_id {item_id}: {df_test_object[col].unique()}')
                    not_shared_str_cols.append(col)
                else:
                    # print(f'Column {col} for item_id {coord_col} have the same value, i.e. we can retain the metadata when grouping by item_id')
                    shared_str_cols.append(col)

    # Check that all shared_str_cols are the same for all item_ids
    shared_str_cols_unique = np.unique(shared_str_cols)
    not_shared_str_cols_unique = np.unique(not_shared_str_cols)

    # Drop the not_shared_str_cols from df_item_id
    df_grouped_final = df_grouped.drop(columns=not_shared_str_cols_unique)

    return df_grouped_final


def get_roi_list_name(rois_of_interest: str):
    """
	If roi exists in d_roi_lists_names, use the name in the dictionary otherwise use the string itself.
	E.g. if we pass rois_of_interest = 'lang_LH_ROIs', we get back the list of ROIs in the language network.
	We do retain the 'lang_LH_ROIs' as the name (for use in save string, etc.)


	:param rois_of_interest:
	:param d_roi_lists_names:
	:return:
	"""

    rois_of_interest_name = rois_of_interest
    if rois_of_interest in d_roi_lists_names.keys():
        rois_of_interest = d_roi_lists_names[rois_of_interest]
    else:
        rois_of_interest = None  # I.e. just use the string as the name, and pass a None such that we use all ROIs

    return rois_of_interest, rois_of_interest_name


def cond_barplot(df: pd.DataFrame,
                 target_UIDs: typing.Union[list, np.ndarray],
                 rois_of_interest: str,
                 x_val: str = 'cond',
                 y_val: str = 'response_target',
                 ylim: typing.Union[list, None] = None,
                 yerr_type: str = 'std_over_UIDs',
                 individual_data_points: typing.Union[str, None] = None,
                 save: bool = False,
                 base_savestr: str = '',
                 add_savestr: str = '',
                 PLOTDIR: typing.Union[str, None] = None,
                 CSVDIR: typing.Union[str, None] = None,
                 ):
    """
	Plot barplots

	Args
		df (pd.DataFrame): Dataframe with rows = items (responses for individual participants) and cols with neural data and condition info.
		target_UIDs (list): List of target UIDs to include in the plot
		rois_of_interest: str, which ROIs to plot (uses the dictionary d_roi_lists_names)
		x_val (str): Which column to use for x-axis
		y_val (str): Which column to use for y-axis
		ylim (list): List of two values, lower and upper limit for y-axis
		yerr_type (str): Type of error bars to use. Options:
			'std_over_items', 'sem_over_items',
			'std_over_UIDs', 'sem_over_UIDs'
			'std_within_UIDs', 'sem_within_UIDs'
		individual_data_points (str): Whether to plot individual data points. Options: 'participant', 'item', None
		save (bool): Whether to save the plot
		base_savestr (str): Base savestr to use
		add_savestr (str): Additional savestr to add to base savestr
		PLTDIR (str): Directory to save plots to
		CSVDIR (str): Directory to save csvs to
	"""
    # Define savestr
    target_UIDs_str = '-'.join(target_UIDs)

    # Filter data
    df = df.copy(deep=True)[df['target_UID'].isin([int(x) for x in target_UIDs])]
    rois_of_interest, rois_of_interest_name = get_roi_list_name(rois_of_interest=rois_of_interest)

    ylim_str = '-'.join([str(x) for x in ylim]) if ylim is not None else 'None'

    for roi in rois_of_interest:
        savestr = f'cond-barplot_' \
                  f'X={x_val}_Y={y_val}_' \
                  f'y={ylim_str}_' \
                  f'YERR={yerr_type}_points={individual_data_points}_' \
                  f'{target_UIDs_str}_{roi}_' \
                  f'{base_savestr}{add_savestr}'

        df_roi = df.copy(deep=True).query(f'roi == "{roi}"')

        # Add error bars
        if yerr_type == 'std_over_items':
            yerr = df_roi.groupby(df_roi[x_val]).std(ddof=1)[y_val]

        elif yerr_type == 'sem_over_items':
            yerr = df_roi.groupby(df_roi[x_val]).sem(ddof=1)[y_val]

        elif yerr_type == 'std_over_UIDs':
            df_temp = df_roi.groupby([x_val, 'target_UID']).mean()

            # Get std over participants for each cond (which is the multiindex)
            yerr = df_temp.groupby(df_temp.index.get_level_values(0)).std(ddof=1)[y_val]

        elif yerr_type == 'sem_over_UIDs':
            df_temp = df_roi.groupby([x_val, 'target_UID']).mean()

            # Get sem over participants for each cond (which is the multiindex)
            yerr = df_temp.groupby(df_temp.index.get_level_values(0)).sem(ddof=1)[y_val]

        elif yerr_type.endswith('within_UIDs'):
            df_temp = df_roi.groupby([x_val, 'target_UID']).mean()

            # Create [subject; cond] pivot table
            df_piv = df_temp.pivot_table(index='target_UID', columns=x_val, values=y_val)

            # Subtract the mean across conditions for each subject
            df_demeaned = df_piv.sub(df_piv.mean(axis=1),
                                     axis=0)  # get a mean value for each subject across all conds, and subtract it from individual conds

            # Take std or sem over subjects
            if yerr_type == 'std_within_UIDs':
                yerr = df_demeaned.std(axis=0, ddof=1)
            elif yerr_type == 'sem_within_UIDs':
                yerr = df_demeaned.sem(axis=0, ddof=1)

        else:
            raise ValueError('yerr_type not recognized')

        print(f'\n\nPlotting: {savestr}\n'
              f'Total number of data points across {len(df_roi.target_UID.unique())} participants: {len(df_roi)}')

        df_cond = df_roi.groupby(df_roi[x_val]).mean()
        # Reindex such that D, S, B
        df_cond = df_cond.reindex(['D', 'S', 'B'])
        yerr = yerr.reindex(['D', 'S', 'B'])

        assert (yerr.index == df_cond.index).all()

        # Plot min/max values in d_cond (take into account +- yerr)
        print(f'Min value: {df_cond[y_val].min() - yerr.max():.2f}')
        print(f'Max value: {df_cond[y_val].max() + yerr.max():.2f}')

        # Plot with condition colors
        color_order = [d_colors[x] for x in df_cond.index.values]

        plt.figure(figsize=(5, 8))
        plt.bar(df_cond.index,
                df_cond[y_val],
                yerr=yerr,
                color=color_order,
                zorder=0,
                error_kw=dict(lw=2))
        if individual_data_points is not None:
            if individual_data_points == 'item':
                jitter = 0.1
                for i, cond in enumerate(df_cond.index):
                    plt.scatter(
                        np.random.uniform(i - jitter, i + jitter, len(df_roi.loc[df_roi[x_val] == cond, :].target_UID)),
                        df_roi.loc[df_roi[x_val] == cond, :][y_val],
                        s=40,
                        color='black',  # color_order[i],
                        zorder=10,
                        edgecolors='none',
                        alpha=0.3)
            elif individual_data_points == 'UID':
                jitter = 0.05
                for i, cond in enumerate(df_cond.index):
                    plt.scatter(np.random.uniform(i - jitter, i + jitter,
                                                  len(df_roi.loc[df_roi[x_val] == cond, :].target_UID.unique())),
                                df_roi.groupby([x_val, 'target_UID']).mean().loc[
                                df_roi.groupby([x_val, 'target_UID']).mean().index.get_level_values(0) == cond, :][
                                    y_val],
                                s=90,
                                color='black',  # color_order[i]
                                edgecolors='none',
                                zorder=10,
                                alpha=0.4)

        # Make ticks and axis labels bigger
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=18)
        plt.xlabel(d_axes_legend[x_val], fontsize=20)
        plt.ylabel(f'{d_axes_legend[y_val]} (mean ± {d_axes_legend[yerr_type]})', fontsize=18)
        plt.title("\n".join(wrap(savestr, 33)), fontsize=14)
        if ylim is not None:
            plt.ylim(ylim)
        plt.tight_layout()
        if save:
            savestr = shorten_savestr(savestr=savestr)
            # Save plot
            os.chdir(PLOTDIR)  # Avoid too long path names
            plt.savefig(savestr + '.pdf', dpi=180)
            # Save data
            df_cond[f'yerr_{yerr_type}'] = yerr  # Add in the yerr values to df_cond
            os.chdir(CSVDIR)
            df_cond.to_csv(join(CSVDIR, savestr + '.csv'))
        plt.show()

def condlevel_perc_inc(df: pd.DataFrame,
                       target_UIDs: typing.Union[list, np.ndarray],
                       response_target_col: str = 'response_target_non_norm',
                       rois_of_interest: str = 'lang_LH_netw',
                       perc_inc_from_cond: str = 'B',
                       compare_to_conds: typing.Union[list, np.ndarray] = ['D', 'S'],
                        save: bool = False,
                        base_savestr: str = '',
                        add_savestr: str = '',
                        CSVDIR: str = '',):
    """
    Quantify the percent increase based on condition averages.
    Performs this computation on the data that is averaged across target_UIDs (doesn't matter if we go straight from
    trial-level or from averaged across participants trial-level data to compute the condition averages).

    Args
        df (pd.DataFrame): Dataframe with rows as items and column with response_target
        target_UIDs (list): List of target_UIDs to include in the analysis
        response_target_col (str): Column with the response target
        rois_of_interest (str): ROI to include in the analysis
        perc_inc_from_cond (str): Condition to compute the percent increase from
        compare_to_conds (list): List of conditions to compare to
        save (bool): Whether to save the plot
        base_savestr (str): Base savestr to add to the plot
        add_savestr (str): Additional savestr to add to the plot
        CSVDIR (str): Directory to save the csv

    Returns
        df_cond with the percent increase computed across condition means

    """

    # Define savestr
    target_UIDs_str = '-'.join(target_UIDs)

    # Filter data
    df_target = df.copy(deep=True)[df['target_UID'].isin([int(x) for x in target_UIDs])]
    rois_of_interest, rois_of_interest_name = get_roi_list_name(rois_of_interest=rois_of_interest)

    for roi in rois_of_interest:

        savestr = f'condlevel-perc-inc_' \
                  f'from-cond={perc_inc_from_cond}_' \
                  f'{target_UIDs_str}_{response_target_col}_{roi}_' \
                  f'{base_savestr}{add_savestr}'

        df_roi = df_target.copy(deep=True).query(f'roi == "{roi}"')

        print(f'\n\nPlotting: {savestr}\n'
              f'Total number of data points across {len(df_roi.target_UID.unique())} participants: {len(df_roi)}')

        # Create df that is grouped by item id. Keep string columns as is (we can perform the groupby cond directly too -- does not matter
        # as long as we have the same number of items per participant)
        df_item_id = groupby_coord(df=df_roi, coord_col='item_id', aggfunc='mean',)

        # Get mean of the conditions
        df_cond = df_item_id.groupby('cond').mean()
        df_cond_from = df_cond.loc[df_cond.index == perc_inc_from_cond]

        # Get perc increase from perc_inc_from_cond condition

        # Compute percent increase from mean of perc_inc_from_cond condition to the mean of the other conditions (compare_to_conds)
        for compare_to_cond in compare_to_conds:
            df_cond_to = df_cond.loc[df_cond.index == compare_to_cond]
            inc = (df_cond_to[response_target_col].values - df_cond_from[response_target_col].values) / df_cond_from[response_target_col].values * 100
            print(f'Percent increase from {perc_inc_from_cond} to {compare_to_cond}: {inc[0]:.3f}%')

            # Add to df_cond
            df_cond.loc[compare_to_cond, f'perc_inc_from_{perc_inc_from_cond}'] = inc[0]

        if save:
            savestr = shorten_savestr(savestr=savestr)

            # Store df_cond as csv
            os.chdir(CSVDIR)
            df_cond.to_csv(f'{savestr}.csv')

        return df_cond

def condlevel_perc_inc_blocked(df: pd.DataFrame,
                            target_UIDs: typing.Union[list, np.ndarray],
                           response_target_col: str = 'response_target_non_norm',
                           rois_of_interest: str = 'lang_LH_netw',
                            perc_inc_from_cond: str = 'B',
                            compare_to_conds: typing.Union[list, np.ndarray]=['D', 'S'],
                            save: bool = False,
                            base_savestr: str = '',
                            add_savestr: str = '',
                            CSVDIR: str = '',):
    """
    Quantify the percent increase based on condition averages. Same as condlevel_perc_inc but for blocked data.
    (no item_ids, and neural data is called response and not response_target)

    Args
        df (pd.DataFrame): Dataframe with rows as items and column with response
        target_UIDs (list): List of target_UIDs to include in the analysis (of str numbers)
        response_target_col (str): Column with the response target
        rois_of_interest (str): ROI to include in the analysis
        perc_inc_from_cond (str): Condition to compute the percent increase from
        compare_to_conds (list): List of conditions to compare to
        save (bool): Whether to save the plot
        base_savestr (str): Base savestr to add to the plot
        add_savestr (str): Additional savestr to add to the plot
        CSVDIR (str): Directory to save the csv

    Returns
        df_cond with the percent increase computed across condition means

    """

    # Define savestr
    target_UIDs_str = '-'.join(target_UIDs)

    # Filter data
    df_target = df.copy(deep=True)[df['target_UID'].isin([int(x) for x in target_UIDs])]
    rois_of_interest, rois_of_interest_name = get_roi_list_name(rois_of_interest=rois_of_interest)

    for roi in rois_of_interest:

        savestr = f'condlevel-perc-inc-blocked_' \
                  f'from-cond={perc_inc_from_cond}_' \
                  f'{target_UIDs_str}_{response_target_col}_{roi}_' \
                  f'{base_savestr}{add_savestr}'

        df_roi = df_target.copy(deep=True).query(f'roi == "{roi}"')

        print(f'\n\nPlotting: {savestr}\n'
              f'Total number of data points across {len(df_roi.target_UID.unique())} participants: {len(df_roi)}')

        # Get mean of the conditions
        df_cond = df_roi.groupby('cond').mean()
        df_cond_from = df_cond.loc[df_cond.index == perc_inc_from_cond]

        # Get perc increase from perc_inc_from_cond condition

        # Compute percent increase from mean of perc_inc_from_cond condition to the mean of the other conditions (compare_to_conds)
        for compare_to_cond in compare_to_conds:
            df_cond_to = df_cond.loc[df_cond.index == compare_to_cond]
            inc = (df_cond_to[response_target_col].values - df_cond_from[response_target_col].values) / df_cond_from[response_target_col].values * 100
            print(f'Percent increase from {perc_inc_from_cond} to {compare_to_cond}: {inc[0]:.3f}%')

            # Add to df_cond
            df_cond.loc[compare_to_cond, f'perc_inc_from_{perc_inc_from_cond}'] = inc[0]

        if save:
            savestr = shorten_savestr(savestr=savestr)

            # Store df_cond as csv
            os.chdir(CSVDIR)
            df_cond.to_csv(f'{savestr}.csv')

        return df_cond


def item_scatter(df: pd.DataFrame,
                 target_UIDs: typing.Union[list, np.ndarray],
                 rois_of_interest: str,
                 x_val: str = f'encoding_model_pred',
                 y_val: str = 'response_target',
                 yerr_type: typing.Union[str, None] = 'sem_over_UIDs',
                 add_mean: bool = False,
                 xlim: typing.Union[list, np.ndarray, None] = None,
                 ylim: typing.Union[list, np.ndarray, None] = None,
                 plot_aspect: float = 1,
                 add_identity: bool = True,
                 save: bool = False,
                 base_savestr: str = '',
                 add_savestr: str = '',
                 PLOTDIR: typing.Union[str, None] = None,
                 CSVDIR: typing.Union[str, None] = None,
                 ):
    """
	Plot scatter per item.

	Args
		df (pd.DataFrame): Dataframe with rows = items (responses for individual participants) and cols with neural data and condition info.
		target_UIDs (list): List of target UIDs to include in the plot
		rois_of_interest: str, which ROIs to plot (uses the dictionary d_roi_lists_names)
		x_val (str): Which column to use for the x-axis (Predicted response intended:
			'encoding_model_pred')
		y_val (str): Which column to use for the y-axis (Actual response intended)
		yerr_type (bool): which y error to compute, options are:
			'sem_over_UIDs': compute sem (ddof=1) across participants for a given item
		add_mean (bool): whether to add a mean line to the plot of the conditions
		xlim (list): x-axis limits
		ylim (list): y-axis limits
		plot_aspect (float): aspect ratio of the plot
		add_identity (bool): whether to add an identity line to the plot, x=y
		base_savestr (str): Base string to use for saving the plot
		add_savestr (str): Additional string to use for saving the plot
		save (bool): Whether to save the plot)"""

    # Define savestr
    target_UIDs_str = '-'.join(target_UIDs)

    # Filter data
    df = df.copy(deep=True)[df['target_UID'].isin([int(x) for x in target_UIDs])]
    rois_of_interest, rois_of_interest_name = get_roi_list_name(rois_of_interest=rois_of_interest)

    for roi in rois_of_interest:

        # If xlim and ylim are not None, join them into a string
        if xlim is not None:
            xlim_str = f'{xlim[0]}-{xlim[1]}'
        else:
            xlim_str = 'None'
        if ylim is not None:
            ylim_str = f'{ylim[0]}-{ylim[1]}'
        else:
            ylim_str = 'None'

        savestr = f'item-scatter_' \
                  f'X={x_val}_Y={y_val}_' \
                  f'YERR={yerr_type}_' \
                  f'xl={xlim_str}_yl={ylim_str}_' \
                  f'mean={add_mean}_' \
                  f'a={plot_aspect}_' \
                  f'i={add_identity}_' \
                  f'{target_UIDs_str}_{roi}_' \
                  f'{base_savestr}{add_savestr}'

        df_roi = df.copy(deep=True).query(f'roi == "{roi}"')

        print(f'\n\nPlotting: {savestr}\n'
              f'Total number of data points across {len(df_roi.target_UID.unique())} participants: {len(df_roi)}')

        # Create df that is grouped by item id. Keep string columns as is
        df_item_id = groupby_coord(df=df_roi, coord_col='item_id', aggfunc='mean', )

        # Get yerr
        if yerr_type is not None:
            if yerr_type == 'sem_over_UIDs':
                df_yerr = groupby_coord(df=df_roi, coord_col='item_id', aggfunc='sem', )

        # Scatterplot
        color_order = [d_colors[x] for x in df_item_id['cond'].values]

        df_x = df_item_id.copy(deep=True).loc[:, [x_val]]

        assert (df_x.index == df_item_id.index).all(), 'Index of df_x and df_item_id do not match'

        if add_mean:  # We want to compute the means of the conditions and add them to the plot as horizontal lines
            df_mean = df_item_id.groupby('cond').mean().loc[:, [x_val, y_val]]

        # Print min/max of x and y (to ensure all data points are within the plot limits)
        print(f'Min of {x_val}: {df_x[x_val].min():2f} and max of {x_val}: {df_x[x_val].max():2f}')
        print(f'Min of {y_val}: {df_item_id[y_val].min():2f} and max of {y_val}: {df_item_id[y_val].max():2f}')

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_box_aspect(plot_aspect)
        ax.scatter(df_x[x_val],  # Predicted
                   df_item_id[y_val],  # Actual
                   c=color_order,
                   alpha=0.6,
                   edgecolors='none')
        if yerr_type is not None:
            ax.errorbar(df_x[x_val],  # Predicted
                        df_item_id[y_val],  # Actual
                        yerr=df_yerr[y_val],
                        fmt='none',
                        # use color_order for the error bars as well
                        ecolor=color_order,
                        alpha=0.7,
                        linewidth=0.4)
        if add_mean:
            for cond in df_mean.index:
                ax.axhline(df_mean.loc[cond, y_val], color=d_colors[cond], linestyle='--', alpha=0.6)
        # Make ticks and axis labels bigger
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if x_val in d_axes_legend:
            plt.xlabel(d_axes_legend[x_val], fontsize=16)
        else:
            plt.xlabel(x_val, fontsize=16)
        if y_val.endswith('noise'):
            plt.ylabel(d_axes_legend[y_val], fontsize=16)  # f'Actual ({target_UIDs_str})'
        else:
            plt.ylabel(f'Actual ({target_UIDs_str})', fontsize=16)
        plt.title("\n".join(wrap(savestr, 60)), fontsize=14)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if add_identity:  # Plot x=y line
            plot_identity(ax, color='black', linestyle='-', alpha=0.6)
        plt.tight_layout()
        # Obtain r value for all values plotted, and for T, D, and B conds separately. Add numbers to plot
        r, p = pearsonr(df_x[x_val], df_item_id[y_val])
        r_B, p_B = pearsonr(df_x.loc[df_item_id['cond'] == 'B', x_val],
                            df_item_id.loc[df_item_id['cond'] == 'B', y_val])
        r_D, p_D = pearsonr(df_x.loc[df_item_id['cond'] == 'D', x_val],
                            df_item_id.loc[df_item_id['cond'] == 'D', y_val])
        r_S, p_S = pearsonr(df_x.loc[df_item_id['cond'] == 'S', x_val],
                            df_item_id.loc[df_item_id['cond'] == 'S', y_val])
        plt.text(0.01, 0.95, f'r={r:.2f}, p={p:.3}', fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.01, 0.90, f'r_T={r_B:.2f}, p_T={p_B:.3}', fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.01, 0.85, f'r_D={r_D:.2f}, p_D={p_D:.3}', fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.01, 0.80, f'r_B={r_S:.2f}, p_B={p_S:.3}', fontsize=10, transform=plt.gca().transAxes)
        if save:
            savestr = shorten_savestr(savestr=savestr)

            os.chdir(PLOTDIR)
            plt.savefig(savestr + '.pdf', dpi=300)

            os.chdir(CSVDIR)
            # Merge df with df_x and save
            df_item_id_store = pd.concat([df_item_id.rename(columns={y_val: f'y_{y_val}'}),
                                          # Ensure that our x and y vals are renamed such that we know what was plotted
                                          df_x.rename(columns={x_val: f'x_{x_val}'})], axis=1)
            # Add a col with the r and p values
            df_item_id_store['r_all-conds'] = r
            df_item_id_store['p_all-conds'] = p
            df_item_id_store['r_B'] = r_B
            df_item_id_store['p_B'] = p_B
            df_item_id_store['r_D'] = r_D
            df_item_id_store['p_D'] = p_D
            df_item_id_store['r_S'] = r_S
            df_item_id_store['p_S'] = p_S

            df_item_id_store.to_csv(savestr + '.csv')
        plt.show()


def NC_across_ROIs(df: pd.DataFrame,
                   rois: typing.Union[list, np.ndarray],
                   nc_col: str = 'nc',
                   nc_err_col: str = 'split_half_se',
                   save: bool = False,
                   ylim: typing.Union[list, np.ndarray] = None,
                   add_space: bool = True,
                   base_savestr: str = '',
                   PLOTDIR: str = '',
                   CSVDIR: str = '', ):
    """
	Plot the NC (y-axis) across ROIs (x-axis)

	Args
		df: pd.DataFrame with index = rois, columns = nc_col, nc_err_col
		rois: list of rois to plot
		nc_col: column name for NC (y-axis)
		nc_err_col: column name for NC error (y-axis)
		save: bool, whether to save the figure
		add_space: bool, whether to add space between the points when networks change
		base_savestr: str, base savestr for the figure
		PLOTDIR: str, directory to save the figure
		CSVDIR: str, directory to save the csv file


	"""
    # check which rois_plot are not in df_nc_roi_all
    rois_not_in_df = [roi for roi in rois if roi not in df.index]
    print(
        f'rois_not_in_nc: {rois_not_in_df}')  # We do not have NC for md_LH_midFrontalOrb (because 848 had neg t-stat for this ROI)
    df_nc_roi = df.loc[[roi for roi in rois if roi not in rois_not_in_df]]

    # All ROIs have a color specified in d_netw_colors
    colors = [d_netw_colors[x] for x in df_nc_roi.index]

    # Make the roi labels nice
    roi_labels = make_ROI_labels_nice(roi_list=df_nc_roi.index.values)
    nc_y = df_nc_roi[nc_col].values
    nc_yerr = df_nc_roi[nc_err_col].values

    if add_space:
        # # Insert a blank (0) between networks (so when Lang switches to MD and then MD switches to DMN)
        replace_flag = False
        for i in range(len(roi_labels)):
            if replace_flag:
                replace_flag = False  # To avoid checking a zero str when we just inserted a blank
                continue
            if i == 0:
                continue
            if roi_labels[i][0] != roi_labels[i - 1][0]:
                replace_flag = True
                roi_labels.insert(i, '')
                nc_y = np.insert(nc_y, i, 0)
                nc_yerr = np.insert(nc_yerr, i, 0)
                colors.insert(i, 'white')

    # Define savestr
    if ylim is not None:
        ylimstr = f'{ylim[0]}-{ylim[1]}'

    savestr = f'NC-across-ROIs_' \
              f'Y={nc_col}_YERR={nc_err_col}_' \
              f'yl={ylimstr}_' \
              f'space={add_space}_' \
              f'n_ROIs={len(df_nc_roi.index)}_' \
              f'{base_savestr}'

    # Y-axis: nc with error split_half_se across all ROIs (index) on x-axis
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(x=np.arange(len(nc_y)), y=nc_y,
                yerr=nc_yerr,
                ls='none',
                ecolor=colors)
    ax.scatter(x=np.arange(len(nc_y)), y=nc_y,
               color=colors,
               marker='o',
               s=50,
               linewidths=0)
    plt.title("\n".join(wrap(savestr, 80)))
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('ROI')
    plt.ylabel('NC + split-half reliability')
    # Make y ticks larger
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=12.5)
    ax.set_xticks(np.arange(len(nc_y)))
    ax.set_xticklabels(roi_labels, rotation=90)
    # Make dotted line for NC = 0
    plt.axhline(y=0, color='k', linestyle='--')
    plt.tight_layout(pad=2)
    if save:
        savestr = shorten_savestr(savestr)
        os.chdir(PLOTDIR)
        plt.savefig(f'{savestr}.pdf', dpi=180)

        # Save the df as csv
        os.chdir(CSVDIR)
        df_nc_roi.to_csv(f'{savestr}.csv')
    plt.show()

    print(f'NC across ROIs: {df_nc_roi[nc_col].values}')


def add_noise_to_pred(df: pd.DataFrame,
                      pred_col: str = 'encoding_model_pred',
                      neural_col: str = 'response_target',
                      pooled_neural_sd: typing.Union[float, None] = None,
                      num_participants: typing.Union[int, None] = None) -> pd.DataFrame:
    """
    Given a column of predicted neural data, and a column of actual neural data, add noise to the predicted neural data
    based on the participant variability in the actual neural data.
    The process is:

    COMPUTATION OF TRUE SD ACROSS PARTICIPANTS:
    For each of 1500 items,
        1. compute sd across 3 subjects (using n-1)
        2. square sd.^2, mean across 1500, sqrt  ==> pooled sd

    SIMULATION OF PARTICIPANT NOISE ON PREDICTIONS:
    Simulate:
    For each of 1500 items,
        simulate y_hat + N(0,sd ~ #2), three times
        mean across those three measurements


    Args:
        df: dataframe with rows as items, and columns containing predicted neural data and actual neural data
        pred_col: column name of predicted neural data, default is 'encoding_model_pred'
        neural_col: column name of actual neural data, default is 'response_target'
        pooled_neural_sd: if not None, use this value as the pooled sd across participants (nice for simulations)
        num_participants: if not None, use this value as the number of participants (nice for simulations)

    Returns:
        df_to_return: input dataframe with column: {pred_col}_noise added to it

    """
    df_to_return = df.copy(deep=True)

    #### First, compute the pooled sd over participants in the real neural data for each item_id (1500) ####
    if not pooled_neural_sd:  # Compute it based on real data
        df_item_id_sd = df_to_return.groupby('item_id')[neural_col].std(ddof=1) # For each item_id, compute sd across participants
        pooled_neural_sd = np.sqrt(np.mean(df_item_id_sd ** 2)) # Taking the average of standard deviations
    else:
        print(f'Using input pooled_neural_sd: {pooled_neural_sd}')  # Manual input

    #### Second, for each item_id, take the predicted neural data, and add N(0, pooled_sd) to it ####
    if not num_participants:
        num_participants = df_to_return.target_UID.nunique()
    else:
        print(f'Using input num_participants: {num_participants}')

    for item_id in df_to_return.item_id.unique():
        # Get the predicted neural data for this item_id
        df_one_item = df_to_return[df_to_return.item_id == item_id]

        # Add N(0, pooled_neural_sd) to it
        df_one_item_pred = df_one_item[
            pred_col]  # Get the predicted value for that given item (should be the same for all participants)

        if not len(np.unique(df_one_item_pred)) == 1:
            print(f'Warning: predicted neural data for item_id {item_id} is not the same across participants')
        # assert len(np.unique(df_one_item_pred)) == 1  # Assert that vals are the same
        item_pred = df_one_item_pred.iloc[0]

        # Sample noise from N(0, pooled_neural_sd) and add to the predicted value as many times as there are participants
        lst_item_id_pred_noise = []
        for i in range(num_participants):
            # Simulate y_hat + N(0,sd ~ #2), three times
            item_pred_noise = item_pred + np.random.normal(0,
                                                           pooled_neural_sd)  # Just samples one value from the Gaussian distribution
            lst_item_id_pred_noise.append(item_pred_noise)

        # Mean across those three measurements
        item_id_pred_noise_mean = np.mean(lst_item_id_pred_noise)

        # Create new col with this value
        df_to_return.loc[df_to_return.item_id == item_id, f'{pred_col}_noise'] = item_id_pred_noise_mean

    return df_to_return


def heatmap(df_corr: pd.DataFrame = None,
            title: str = None,
            savestr: str = None,
            save: bool = False,
            vmin: float = None,
            vmax: float = None,
            center: float = None,
            pretty_roi_labels: bool = False,
            annot: bool = False,
            figsize: tuple = (20, 20),
            PLOTDIR: str = None,
            CSVDIR: str = None, ):
    """
    Plot heatmap of correlation matrix.

    Args:
        df_corr (pandas.DataFrame): correlation matrix
        title (str): title of the plot
        savestr (str): string to append to the filename
        save (bool): whether to save the plot
        vmin (float): minimum value for the colorbar
        vmax (float): maximum value for the colorbar
        center (float): center value for the colorbar
        pretty_roi_labels (bool): whether to make the labels more interpretable for ROI correlations
        annot (bool): whether to annotate the heatmap with the correlation values
        figsize (tuple): figure size
        PLOTDIR (str): directory to save the plot
        CSVDIR (str): directory to save the csv file

    """

    if pretty_roi_labels:
        # Replace _ with space and uppercase first letter
        df_corr.index = df_corr.index.str.replace('_', ' ')
        df_corr.index = [s[0].upper() + s[1:] for s in df_corr.index]
        df_corr.columns = df_corr.index

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_corr,
                annot=annot,
                fmt='.2f',
                ax=ax,
                cmap='RdBu_r',
                square=True,
                vmin=vmin,
                vmax=vmax,
                center=center,
                cbar_kws={'label': 'Pearson R',
                          'shrink': 0.5,
                          },
                # Make annot bigger
                annot_kws={'size': 14},
                )
    plt.title(title, fontsize=20)
    # Make labels bigger
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(pad=1)
    if save:
        plt.savefig(join(PLOTDIR, savestr + '.pdf'), dpi=180)
        df_corr.to_csv(join(CSVDIR, savestr + '.csv'))
    plt.show()

def feature_heatmap(df_feats: pd.DataFrame,
                    fname_save: str,
                    save: bool = False,
                    annot: bool = True,
                    cmap: str = 'RdBu_r',
                    center: float = 0,
                    lower_triu: bool = True,
                    vmin: float = -1,
                    vmax: float = 1,
                    PLOTDIR: str = '',
                    CSVDIR: str = '', ):
    """
    Plot a heatmap of the correlation between the features in df_feats.
    """

    # Num items: rows
    num_items = len(df_feats)
    num_feats = len(df_feats.columns)

    # Compute correlation matrix
    df_feats_corr = df_feats.corr()

    if lower_triu:
        # only plot lower triangle
        mask = np.triu(np.ones_like(df_feats_corr, dtype=np.bool))
        df_feats_corr = df_feats_corr.mask(mask)

    # Figure out the labels using d_axes_legend
    labels = [d_axes_legend[feat] for feat in df_feats_corr.index]

    savestr = f'feature-heatmap_' \
              f'a={annot}_ce={center}_' \
              f'vmi={vmin}_vma={vmax}_' \
              f'ltri={lower_triu}_' \
              f'n-items={num_items}_n-feats={num_feats}_' \
              f'{fname_save}'

    # Heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_feats_corr,
                annot=annot,
                cmap=cmap,
                center=center,
                vmin=vmin, vmax=vmax,
                annot_kws={'size': 12},
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Pearson correlation coefficient',
                          'shrink': 0.6, },
                )
    plt.xticks(rotation=60, ha='right', fontsize=13)
    plt.yticks(rotation=0, fontsize=13)
    plt.title('\n'.join(wrap(savestr, 60)))
    plt.tight_layout()
    if save:
        savestr = shorten_savestr(savestr=savestr)

        os.chdir(PLOTDIR)
        plt.savefig(f'{savestr}.pdf', dpi=180)

        os.chdir(CSVDIR)
        df_feats_corr.to_csv(f'{savestr}.csv')

    plt.show()


def feat_scatter(df: pd.DataFrame,
                 target_UIDs: typing.Union[list, np.ndarray],
                 x_val: str = f'rating_arousal_mean',
                 y_val: str = 'response_target',
                 yerr_type: typing.Union[str, None] = None,
                 add_best_fit_line: bool = False,
                 xlim: typing.Union[list, np.ndarray, None] = None,
                 ylim: typing.Union[list, np.ndarray, None] = None,
                 plot_aspect: float = 1,
                 save: bool = False,
                 base_savestr: str = '',
                 add_savestr: str = '',
                 PLOTDIR: typing.Union[str, None] = None,
                 CSVDIR: typing.Union[str, None] = None,
                 ):
    """
    Plot scatter per item.

    Args
        df (pd.DataFrame): Dataframe with rows = items (responses for individual participants) and cols with neural data and condition info.
        target_UIDs (list): List of target UIDs to include in the plot
        rois_of_interest: str, which ROIs to plot (uses the dictionary d_roi_lists_names)
        x_val (str): Which column to use for the x-axis (Predicted response intended:
            'encoding_model_pred')
        y_val (str): Which column to use for the y-axis (Actual response intended)
        yerr_type (bool): which y error to compute, options are:
            'sem_over_UIDs': compute sem (ddof=1) across participants for a given item
        add_best_fit_line (bool): whether to add a best fit line to the plot
        xlim (list): x-axis limits
        ylim (list): y-axis limits
        plot_aspect (float): aspect ratio of the plot
        base_savestr (str): Base string to use for saving the plot
        add_savestr (str): Additional string to use for saving the plot
        save (bool): Whether to save the plot)"""

    # Define savestr
    target_UIDs_str = '-'.join(target_UIDs)
    roi = df['roi'].unique()
    assert len(roi) == 1
    roi = roi[0] # ROI is already in the basestr name

    # Filter data
    df = df.copy(deep=True)[df['target_UID'].isin([int(x) for x in target_UIDs])]

    # If xlim and ylim are not None, join them into a string
    if xlim is not None:
        xlim_str = f'{xlim[0]}-{xlim[1]}'
    else:
        xlim_str = 'None'
    if ylim is not None:
        ylim_str = f'{ylim[0]}-{ylim[1]}'
    else:
        ylim_str = 'None'

    savestr = f'feat-scatter_' \
              f'X={x_val}_Y={y_val}_' \
              f'YERR={yerr_type}_' \
              f'xl={xlim_str}_yl={ylim_str}_' \
              f'b-fit={add_best_fit_line}_' \
              f'a={plot_aspect}_' \
              f'{target_UIDs_str}_' \
              f'{base_savestr}{add_savestr}'

    df_roi = df.copy(deep=True).query(f'roi == "{roi}"')

    print(f'\n\nPlotting: {savestr}\n'
          f'Total number of data points across {len(df_roi.target_UID.unique())} participants: {len(df_roi)}')

    # Create df that is grouped by item id. Keep string columns as is
    df_item_id = groupby_coord(df=df_roi, coord_col='item_id', aggfunc='mean',)

    # Get yerr
    if yerr_type is not None:
        if yerr_type == 'sem_over_UIDs':
            df_yerr = groupby_coord(df=df_roi, coord_col='item_id', aggfunc='sem',)

    # Scatterplot
    color_order = [d_colors[x] for x in df_item_id['cond'].values]
    # For cond_approach = B, circle. For cond_approach = D_search and S_search, square; for cond_approach = D_modify and S_modify, triangle
    symbol_order = [d_symbols[x] for x in df_item_id['cond_approach'].values]

    # Print min/max of x and y (to ensure all data points are within the plot limits)
    print(f'Min of {x_val}: {df_item_id[x_val].min():2f} and max of {x_val}: {df_item_id[x_val].max():2f}')
    print(f'Min of {y_val}: {df_item_id[y_val].min():2f} and max of {y_val}: {df_item_id[y_val].max():2f}')

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_box_aspect(plot_aspect)
    # Plot in reverse order such that first items are plotted on top
    for i in range(len(df_item_id))[::-1]:
        ax.scatter(df_item_id[x_val].iloc[i],  # feature
                   df_item_id[y_val].iloc[i],  # neural data
                   c=color_order[i],
                   alpha=0.6,
                   edgecolors='none',
                   # set symbol
                   marker=symbol_order[i],
                   s=50)
    if yerr_type is not None:
        ax.errorbar(df_item_id[x_val],
                    df_item_id[y_val],
                    yerr=df_yerr[y_val],
                    fmt='none',
                    # use color_order for the error bars as well
                    ecolor=color_order,
                    alpha=0.6,
                    linewidth=0.4)
    if add_best_fit_line:  # Compute best line for ALL points and only cond = T
        # Compute best fit line for ALL points (use np.polynomial.polynomial.polyfit)
        coefs = np.polynomial.polynomial.polyfit(df_item_id[x_val], df_item_id[y_val], 1)
        ffit = np.polynomial.polynomial.Polynomial(coefs)
        x = np.linspace(df_item_id[x_val].min(), df_item_id[x_val].max(), 100)
        y = ffit(x)
        ax.plot(x, y, linewidth=4, linestyle='-', color='black', zorder=10)
        # Compute best fit line for cond = T
        df_item_id_B = df_item_id[df_item_id['cond'] == 'B']
        coefs_B = np.polynomial.polynomial.polyfit(df_item_id_B[x_val], df_item_id_B[y_val], 1)
        ffit_B = np.polynomial.polynomial.Polynomial(coefs_B)
        # Use same x_B as for all points
        x_B = np.linspace(df_item_id[x_val].min(), df_item_id[x_val].max(), 100)
        y_B = ffit_T(x_B)
        ax.plot(x_B, y_B, linewidth=4, linestyle='--', color='black', zorder=10)
    # Make ticks and axis labels bigger
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # Only plot a few values on the axes
    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # Make ticks thicker
    ax.tick_params(axis='both', which='major', width=2, length=8)
    if x_val in d_axes_legend:
        plt.xlabel(d_axes_legend[x_val], fontsize=32)
    else:
        plt.xlabel(x_val, fontsize=32)
    if y_val in d_axes_legend:
        plt.ylabel(d_axes_legend[y_val], fontsize=28)
    else:
        plt.ylabel(y_val, fontsize=28)
    # plt.title("\n".join(wrap(savestr, 90)), fontsize=10)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    # Obtain r value for all values plotted, and for B, D, and S conds separately. Add numbers to plot
    r, p = pearsonr(df_item_id[x_val], df_item_id[y_val])
    r_B, p_B = pearsonr(df_item_id.loc[df_item_id['cond'] == 'B', x_val],
                        df_item_id.loc[df_item_id['cond'] == 'B', y_val])
    r_D, p_D = pearsonr(df_item_id.loc[df_item_id['cond'] == 'D', x_val],
                        df_item_id.loc[df_item_id['cond'] == 'D', y_val])
    r_S, p_S = pearsonr(df_item_id.loc[df_item_id['cond'] == 'S', x_val],
                        df_item_id.loc[df_item_id['cond'] == 'S', y_val])
    r_D_search, p_D_search = pearsonr(df_item_id.loc[df_item_id['cond_approach'] == 'D_search', x_val],
                                      df_item_id.loc[df_item_id['cond_approach'] == 'D_search', y_val])
    r_S_search, p_S_search = pearsonr(df_item_id.loc[df_item_id['cond_approach'] == 'S_search', x_val],
                                      df_item_id.loc[df_item_id['cond_approach'] == 'S_search', y_val])
    r_D_modify, p_D_modify = pearsonr(df_item_id.loc[df_item_id['cond_approach'] == 'D_modify', x_val],
                                    df_item_id.loc[df_item_id['cond_approach'] == 'D_modify', y_val])
    r_S_modify, p_S_modify = pearsonr(df_item_id.loc[df_item_id['cond_approach'] == 'S_modify', x_val],
                                    df_item_id.loc[df_item_id['cond_approach'] == 'S_modify', y_val])
    r_DS_search, p_DS_search = pearsonr(
        df_item_id.loc[df_item_id['cond_approach'].isin(['D_search', 'S_search']), x_val],
        df_item_id.loc[df_item_id['cond_approach'].isin(['D_search', 'S_search']), y_val])
    r_DS_modify, p_DS_modify = pearsonr(df_item_id.loc[df_item_id['cond_approach'].isin(['D_modify', 'S_modify']), x_val],
                                      df_item_id.loc[df_item_id['cond_approach'].isin(['D_modify', 'S_modify']), y_val])
    # Add all these outside of the plot NOT IN AXES
    # Outside of axes, add r and p values for each condition
    ax.text(0.79, 0.65, f'r={r:.2f}, p={p:.3}', transform=plt.gcf().transFigure, fontsize=10)
    ax.text(0.79, 0.60, f'r_B={r_B:.2f}, p_B={p_B:.3}', transform=plt.gcf().transFigure, fontsize=10)
    # Plot these r values in the axes, lower right
    ax.text(0.25, 0.08, f'r (n=2,000) = {r:.2f}', transform=ax.transAxes, fontsize=18, )
    ax.text(0.25, 0.02, f'r (n=1,000; grey) = {r_B:.2f}', transform=ax.transAxes, fontsize=18, )
    plt.tight_layout(pad=2)
    if save:
        savestr = shorten_savestr(savestr=savestr)
        os.chdir(PLOTDIR)
        plt.savefig(savestr + '.pdf', dpi=72)
        os.chdir(CSVDIR)
        # Ensure that our x and y vals are renamed such that we know what was plotted
        df_item_id_store = df_item_id.rename(columns={x_val: f'x_{x_val}', y_val: f'y_{y_val}'})
        # Add a col with the r and p values
        df_item_id_store['r_all-conds'] = r
        df_item_id_store['p_all-conds'] = p
        df_item_id_store['r_B'] = r_B
        df_item_id_store['p_B'] = p_B
        df_item_id_store['r_D'] = r_D
        df_item_id_store['p_D'] = p_D
        df_item_id_store['r_S'] = r_S
        df_item_id_store['p_S'] = p_S
        df_item_id_store['r_D_search'] = r_D_search
        df_item_id_store['p_D_search'] = p_D_search
        df_item_id_store['r_S_search'] = r_S_search
        df_item_id_store['p_S_search'] = p_S_search
        df_item_id_store['r_D_modify'] = r_D_modify
        df_item_id_store['p_D_modify'] = p_D_modify
        df_item_id_store['r_S_modify'] = r_S_modify
        df_item_id_store['p_S_modify'] = p_S_modify
        df_item_id_store['r_DS_search'] = r_DS_search
        df_item_id_store['p_DS_search'] = p_DS_search
        df_item_id_store['r_DS_modify'] = r_DS_modify
        df_item_id_store.to_csv(savestr + '.csv')
    plt.show()

def get_bin_range(df: pd.DataFrame,
                  feat: str,
                  enforce_bin_edges: bool = False,):
    """
    Find the min and max of the feature and return the feature binned into 6! bins.

    Args
        df (pd.DataFrame): Dataframe with feature (as column) and rows as items
        feat (str): Name of feature to bin
        enforce_bin_edges (bool): If True, then the min and max of the feature will be enforced to min 1 and max 7
            This is to ensure that for the behavioral rating features (spanning 1-7), if the max is e.g., 6.5, we still
            want the max to be 7. T
    """
    df_original = df.copy(deep=True)

    unique_rating_vals = df[feat].unique()

    if enforce_bin_edges:
        min_val = 1
        max_val = 7
    else:
        min_val = df[feat].min()
        max_val = df[feat].max()

    range = max_val - min_val
    bin_size = range / 6

    print(f'Feature: {feat} with min: {min_val} and max: {max_val} and range: {range} and bin_size: {bin_size} (total bins: 6)')
    print(f'Number of unique values: {len(unique_rating_vals)}')

    for i, val in enumerate(unique_rating_vals):
        # If val between 1 and 2, put in bin 1
        if val >= min_val and val < min_val + bin_size:
            df.loc[df[feat] == val, f'{feat}_bin'] = 1
        elif val >= min_val + bin_size and val < min_val + (2 * bin_size):
            df.loc[df[feat] == val, f'{feat}_bin'] = 2
        elif val >= min_val + (2 * bin_size) and val < min_val + (3 * bin_size):
            df.loc[df[feat] == val, f'{feat}_bin'] = 3
        elif val >= min_val + (3 * bin_size) and val < min_val + (4 * bin_size):
            df.loc[df[feat] == val, f'{feat}_bin'] = 4
        elif val >= min_val + (4 * bin_size) and val < min_val + (5 * bin_size):
            df.loc[df[feat] == val, f'{feat}_bin'] = 5
        elif val >= min_val + (5 * bin_size) and val <= np.round(min_val + (6 * bin_size), 13):  # <= because we want to include the max value
            # The np.round is because a small imprecision due to 6 * that causes the min value to not be assigned anywhere.
            df.loc[df[feat] == val, f'{feat}_bin'] = 6
        else:
            print(f'Val {val} not in any bin')

    # Print the bin edges
    round_to = 1
    for bin_idx in np.arange(1, 7, 1):
        print(f'Bin {bin_idx} edges: {min_val + ((bin_idx - 1) * bin_size):.{round_to}f} - {min_val + (bin_idx * bin_size):.{round_to}f}')

    return df


def binned_feat(df: pd.DataFrame,
                x_val_feat: str,
                y_val: str = 'response_target',
                yerr_type: typing.Union[str, None] = 'sem',
                ind_points: bool = False,
                ylim: typing.Union[list, np.ndarray, None] = None,
                plot_aspect: typing.Union[float, None] = None,
                custom_cmap: typing.Union[tuple, None] = None,
                save: bool = False,
                base_savestr: str = '',
                add_savestr: str = '',
                min_trials: int = 20, # 1%
                PLOTDIR: str = None,
                CSVDIR: str = None,
                ):
    """
    Plot mean response target for each bin of a feature.

    Args
        df (pd.DataFrame): DataFrame with colum {feat}_bin and response_target (rows are items, for this case,
            the data is averaged across UIDs. df_item_id
        x_val_feat (str): Name of feature.
        y_val (str): Name of y-axis (response target)
        yerr_type (str): options: 'sem', 'std', None. If 'sem' or 'std', then error bars are plotted (since df has points
            across items, the error will be across items)
        ind_points (bool): If True, then plot individual points (items)
        custom_cmap (tuple): Tuple of (cmap, norm) to use for plotting. Plots each bin with a different color. If None, then
            plot in black.
        save (bool): If True, then save plot
        base_savestr (str): Base string to save plot
        add_savestr (str): Additional string to save plot
        min_trials (int): Minimum number of trials in a bin to plot. If less than this number, then the bin is not plotted (set to nan)
        PLOTDIR (str): Directory to save plot
        CSVDIR (str): Directory to save csv

    Returns
        df_agg_feat (pd.DataFrame): DataFrame with mean, median, std, sem, and count for each bin
    """
    df_item_bins = df.copy(deep=True)

    # Aggregate by bin
    # Groupby feature value (the rating). We get the mean of response target (neural) for each rating value.
    # Aggregate such that we get mean, median, std, sem, and count for each bin
    df_agg_feat = df_item_bins.groupby(f'{x_val_feat}_bin').agg(
        {y_val: ['mean', 'median', 'std', 'sem', 'count']})

    # print min/max of count
    print(f'Min number of data points in a bin: {df_agg_feat[y_val]["count"].min()}')
    print(f'Max number of data points in a bin: {df_agg_feat[y_val]["count"].max()}')

    # If min val is less than min_trials, then set to y_val mean,median,std,sem
    if df_agg_feat[y_val]["count"].min() < min_trials:
        print(f'Number of bins with less than {min_trials} trials: {len(df_agg_feat.loc[df_agg_feat[y_val]["count"] < min_trials])}. '
              f'Setting to 0')
        df_agg_feat.loc[df_agg_feat[y_val]["count"] < min_trials, y_val] = np.nan

    # Find xlim (min - 0.3 and max of feature + 0.3)
    xlim = [df_agg_feat.index.min() - 0.3, df_agg_feat.index.max() + 0.3]

    if ylim is not None:
        ylim_str = f'{ylim[0]}-{ylim[1]}'
    else:
        ylim_str = 'None'

    if custom_cmap is not None:
        cmap_str = [True if custom_cmap is not None else False][0]
    else:
        cmap_str = False

    savestr = f'binned-feat_' \
              f'X={x_val_feat}_Y={y_val}_' \
              f'YERR={yerr_type}_' \
              f'yl={ylim_str}_' \
              f'a={plot_aspect}_' \
              f'ind={ind_points}_' \
              f'mt={min_trials}_' \
              f'c={cmap_str}_' \
              f'{base_savestr}{add_savestr}'

    if yerr_type is not None:
        yerr = df_agg_feat[y_val][yerr_type]
    else:
        yerr = None


    if custom_cmap is not None:
        # Get the "count" from df_agg_feat
        cmap, norm = custom_cmap[0], custom_cmap[1]
        colors_count = df_agg_feat[y_val]['count'].values
        # Map to the passed scale
        colors = cmap(norm(colors_count))
        colors = [matplotlib.colors.rgb2hex(c) for c in colors]


    # Plot mean response target for each bin of a feature
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_box_aspect(plot_aspect)
    if custom_cmap is not None:
        for i in range(len(df_agg_feat.index)): # Plot each value in its respective color
            ax.errorbar(df_agg_feat.index.values[i],
                        df_agg_feat[y_val]['mean'].values[i],
                        yerr=[yerr.values[i] if yerr is not None else None][0], # If yerr is None, then set to None, else fetch the value
                        fmt='o', color=colors[i],
                        markersize=23,
                        elinewidth=3)
        # Also plot the lines
        ax.plot(df_agg_feat.index,
                df_agg_feat[y_val]['mean'],
                color='black', linewidth=3)
        # show the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
    else:
        ax.errorbar(df_agg_feat.index,
                    df_agg_feat[y_val]['mean'],
                    yerr=yerr,
                    fmt='o-', color='black',
                    markersize=23)
    # Still show all bins, even if there are no trials in that bin
    ax.set_xlim(xlim)
    if ind_points:
        ax.scatter(df_item_bins[f'{x_val_feat}_bin'],
                   df_item_bins[y_val],
                   alpha=0.3, s=15, edgecolors='none',
                     color='black')
    ax.set_xlabel(f'{d_axes_legend[x_val_feat]} bin', fontsize=36)
    ax.set_ylabel(d_axes_legend[y_val], fontsize=36)
    # ax.set_title("\n".join(wrap(f'{savestr}', 80)))
    # Make ticks and labels larger
    ax.tick_params(axis='both', which='major', labelsize=36)
    # Only plot a few values on the axes
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # Make ticks thicker
    ax.tick_params(axis='both', which='major', width=2, length=8)
    # If ylim is [-0.5, 0.5], then show ticks at -0.5, 0, 0.5
    if ylim is not None:
        if ylim[0] == -0.5 and ylim[1] == 0.5:
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        if ylim[0] == -2 and ylim[1] == 2:
            ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.tight_layout()
    if ylim is not None:
        ax.set_ylim(ylim)
    if save:
        savestr = shorten_savestr(savestr=savestr)
        os.chdir(PLOTDIR)
        plt.savefig(os.path.join(PLOTDIR, f'{savestr}.pdf'), dpi=180)

        os.chdir(CSVDIR)
        # Store both individual points and grouped data

        df_item_bins.to_csv(os.path.join(CSVDIR, f'{savestr}_item-level.csv'))
        df_agg_feat.to_csv(os.path.join(CSVDIR, f'{savestr}_bin-level.csv'))

    plt.show()

    return df_agg_feat


def binned_feat_pairwise_stats(df: pd.DataFrame,
                               x_val_feat: str,
                               y_val: str = 'response_target',
                               save: bool = False,
                               base_savestr: str = '',
                               add_savestr: str = '',
                               min_trials: int = 20,  # 1%
                               PLOTDIR: str = None,
                               CSVDIR: str = None,
                               ):
    """
    Run pairwise statistical tests between values (y_val) in bins (independent samples) of x_val_feat.
    Note that running this statistics function requires statsmodels (not a part of the core drive-suppress env).

    Args:
        df (pd.DataFrame): DataFrame with colum {feat}_bin and response_target (rows are items, for this case,
            the data is averaged across UIDs. df_item_id
        x_val_feat (str): Name of feature.
        y_val (str): Name of y-axis (response target)
        save (bool): If True, then save plot
        base_savestr (str): Base string to save plot
        add_savestr (str): Additional string to save plot
        min_trials (int): Minimum number of trials in a bin to plot. If less than this number, then the bin is not plotted (set to nan)
        PLOTDIR (str): Directory to save plot
        CSVDIR (str): Directory to save csv
    """

    from scipy.stats import ttest_ind
    import itertools
    from statsmodels.stats.multitest import multipletests

    df_item_bins = df.copy(deep=True)

    savestr = f'binned-feat_' \
              f'X={x_val_feat}_Y={y_val}_' \
              f'{base_savestr}{add_savestr}'

    # Aggregate by bin
    # Groupby feature value (the rating). We get the mean of response target (neural) for each rating value.
    # Aggregate such that we get mean, median, std, sem, and count for each bin
    df_agg_feat = df_item_bins.groupby(f'{x_val_feat}_bin').agg(
        {y_val: ['mean', 'median', 'std', 'sem', 'count']})

    # If min val is less than min_trials, then drop it
    if df_agg_feat[y_val]["count"].min() < min_trials:
        print(f'Number of bins with less than {min_trials} trials: {len(df_agg_feat.loc[df_agg_feat[y_val]["count"] < min_trials])}. '
              f'Dropping these bins: {df_agg_feat.loc[df_agg_feat[y_val]["count"] < min_trials].index.values}')
        df_agg_feat = df_agg_feat.loc[df_agg_feat[y_val]["count"] >= min_trials]

    # Run the unique bin values
    bins = df_agg_feat.index.values
    # Get the unique bin combos
    bin_combos = list(itertools.combinations(bins, 2))
    # In each tuple, switch the order of the bin such that we plot the lower triangle
    bin_combos = [(bin2, bin1) for bin1, bin2 in bin_combos]

    # Run t-test for each bin combo and then plot as a lower triangle heatmap

    # Instantiate a square dataframe with nans
    df_pvals = pd.DataFrame(data=np.nan, index=bins, columns=bins)

    for bin1, bin2 in bin_combos:
        t, p = ttest_ind(df_item_bins.loc[df_item_bins[f'{x_val_feat}_bin'] == bin1, y_val],
                         df_item_bins.loc[df_item_bins[f'{x_val_feat}_bin'] == bin2, y_val])

        # Add to dataframe
        df_pvals.loc[bin1, bin2] = p

    # Plot the pvals as a heatmap
    # Use a cmap where low values are green and high values are red
    cmap = 'RdYlGn_r'

    # Convert index and columns to integers
    df_pvals.index = df_pvals.index.astype(int)
    df_pvals.columns = df_pvals.columns.astype(int)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(df_pvals,
                annot=True,
                ax=ax, cmap=cmap,
                vmin=0, vmax=1,
                square=True,
                cbar_kws={"shrink": .5},
                annot_kws={"size": 15},)
    ax.set_title(f'Uncorrected stats: {d_axes_legend[x_val_feat]}',
                 fontsize=20)
    ax.set_xlabel(f'{d_axes_legend[x_val_feat]} bin', fontsize=15)
    ax.set_ylabel(f'{d_axes_legend[x_val_feat]} bin', fontsize=15)
    # Make ticks and tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    if save:
        # Replace 'binned_feat' with 'binned_feat_uncorr-stats_'
        savestr_uncor = savestr.replace('binned-feat', 'binned-feat_uncorr-stats_')
        savestr_uncor = shorten_savestr(savestr=savestr_uncor)
        os.chdir(PLOTDIR)
        plt.savefig(f'{savestr_uncor}.pdf', dpi=300)

        # Save the pvals as a csv
        os.chdir(CSVDIR)
        df_pvals.to_csv(f'{savestr_uncor}_pvals.csv')

        # Also save df_item_bins as a csv (just this one, for correct it is obv the same)
        df_item_bins.to_csv(f'{savestr_uncor}_df_item_bins.csv')

    plt.show()

    # Correct using benjamini hochberg and add to a new dataframe
    df_pvals_bh = df_pvals.copy(deep=True)
    mask = np.tril(np.ones(df_pvals_bh.shape), k=-1).astype(np.bool)
    p_vals_masked = df_pvals_bh.values[mask]

    # Correct using benjamini hochberg
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_vals_masked, alpha=0.05, method='fdr_bh')

    # Add to dataframe
    df_pvals_bh.values[mask] = pvals_corrected

    # Plot the corrected pvals as a heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(df_pvals_bh,
                annot=True,
                ax=ax, cmap=cmap,
                vmin=0, vmax=1,
                square=True,
                cbar_kws={"shrink": .5},
                annot_kws={"size": 15},)
    ax.set_title(f'FDR-corrected BH: {d_axes_legend[x_val_feat]}',
                 fontsize=20)
    ax.set_xlabel(f'{d_axes_legend[x_val_feat]} bin', fontsize=15)
    ax.set_ylabel(f'{d_axes_legend[x_val_feat]} bin', fontsize=15)
    # Make ticks and tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    if save:
        savestr_bh = savestr.replace('binned-feat', 'binned-feat_bh-corr-stats_')
        savestr_bh = shorten_savestr(savestr=savestr_bh)
        os.chdir(PLOTDIR)
        plt.savefig(f'{savestr_bh}.pdf', dpi=300)

        # Save the pvals as a csv
        os.chdir(CSVDIR)
        df_pvals_bh.to_csv(f'{savestr_bh}_pvals.csv')

    plt.show()


def make_ROI_labels_nice(roi_list: typing.Union[list, np.ndarray],
                         ):
    """Make the ROI labels nicer by:
	Upper case lh and rh
	Replace _ with space
	Uppercase lang to Lang

	Args
		roi_list (list): List of ROI labels
	"""
    roi_nice = []
    for roi in roi_list:
        roi_nice.append(
            roi.replace('lh', 'LH').replace('rh', 'RH').replace('_', ' ').replace('lang', 'Lang').replace('md',
                                                                                                          'MD').replace(
                'dmn', 'DMN'))
    return roi_nice


def plot_identity(axes, *line_args, **line_kwargs):
    """
	Thanks to https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
	"""
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes
