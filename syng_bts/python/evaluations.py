import umap.umap_ as umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
import os

import importlib.resources as pkg_resources


def heatmap_eval(dat_real,dat_generated=None, save = False):
    r"""
    This function creates a heatmap visualization comparing the generated data and the real data.
    dat_generated is applicable only if 2 sets of data is available.
    
    Parameters
    -----------
    dat_real: pd.DataFrame
            the original copy of the data
    dat_generated : pd.DataFrame, optional
            the generated data
    save: bool, optional
            if save = True, return figures
    """
    if dat_generated is None:
        # Only plot dat_real if dat_generated is None
        fig = plt.figure(figsize=(6, 6))
        ax = sns.heatmap(dat_real, cbar=True)
        ax.set_title('Real Data')
        ax.set_xlabel('Features')
        ax.set_ylabel('Samples')
    else:
        # Plot both dat_generated and dat_real side by side
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6),
                                gridspec_kw=dict(width_ratios=[0.5, 0.55]))

        sns.heatmap(dat_generated, ax=axs[0], cbar=False)
        axs[0].set_title('Generated Data')
        axs[0].set_xlabel('Features')
        axs[0].set_ylabel('Samples')

        sns.heatmap(dat_real, ax=axs[1], cbar=True)
        axs[1].set_title('Real Data')
        axs[1].set_xlabel('Features')
        axs[1].set_ylabel('Samples')

    plt.tight_layout()

    if save:
        return fig
    else:
        plt.show()


def UMAP_eval(dat_generated, dat_real, groups_generated=None, groups_real=None, random_state=42, legend_pos="best"):
    r"""
    This function creates a UMAP visualization comparing the generated data and the real data.
    If only 1 set of data is available, dat_generated and groups_generated should have None as inputs.

    Parameters
    -----------
    dat_generated : pd.DataFrame or None
            the generated data, input None if unavailable
    dat_real: pd.DataFrame
            the original copy of the data
    groups_generated : pd.Series or None
            the groups generated, input None if unavailable
    groups_real : pd.Series or None
            the real groups, input None if unavailable
    legend_pos : string
            legend location
    
    """

    if dat_generated is None:
        # Only plot the real data
        reducer = UMAP(random_state=random_state)
        embedding = reducer.fit_transform(dat_real.values)

        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        
        plt.figure(figsize=(10, 8))
        if groups_real is not None:
            umap_df['Group'] = groups_real.astype(str)  # Ensure groups are hashable for seaborn
            sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', style='Group', palette='bright')
            plt.legend(title='Group', loc=legend_pos)
            plt.title('UMAP Projection of Real Data with Groups')
        else:
            plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], alpha=0.7)
            plt.title('UMAP Projection of Real Data')

        plt.show()
        return
    
    # If dat_generated is provided, we process both real and generated data
    # Filter out features with zero variance in generated data
    non_zero_var_cols = dat_generated.var(axis=0) != 0

    # Use loc to filter columns by the non_zero_var_cols boolean mask
    dat_real = dat_real.loc[:, non_zero_var_cols]
    dat_generated = dat_generated.loc[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((dat_real.values, dat_generated.values))  
    combined_labels = np.array(['Real'] * dat_real.shape[0] + ['Generated'] * dat_generated.shape[0])

    # UMAP dimensionality reduction
    reducer = UMAP(random_state=random_state)
    embedding = reducer.fit_transform(combined_data)

    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['Data Type'] = combined_labels

    plt.figure(figsize=(10, 8))

    if groups_real is not None and groups_generated is not None:
        # If group information is available, use it for coloring
        combined_groups = np.concatenate((groups_real, groups_generated))
        combined_groups = [str(group) for group in combined_groups]  # Convert groups to string if not already
        umap_df['Group'] = combined_groups
        sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Data Type', style='Group', palette='bright')
        plt.legend(title='Data Type/Group', loc="best")
        plt.title('UMAP Projection of Real and Generated Data with Groups')

    else:
        # If no group information, just plot real vs. generated data
        sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Data Type', palette='bright')
        plt.legend(title='Data Type', loc="best")
        plt.title('UMAP Projection of Real and Generated Data')

    plt.show()


def evaluation(generated_input: str = "BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv", 
               real_input: str = "BRCASubtypeSel_test.csv"):
    r"""
    This method provides preprocessing of the input data prior to creating the visualizations.
    This can also be used as inspiration for other ways of using the above evaluation methods.

    generated_input : string
        the generated dataset; a default set is also provided as an example
    real_input : string
        the real original dataset; a default set is also provided as an example
    
    """
    train_path = "../Case/BRCASubtype/" + generated_input
    if generated_input == 'BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv' and not os.path.exists(path=train_path):
        with pkg_resources.open_text('syng_bts.Case.BRCASubtype', 'BRCASubtypeSel_train_epoch285_CVAE1-20_generated.csv') as data_file:
            generated = pd.read_csv(data_file)
    else:
        generated = pd.read_csv(train_path, header = 0)
    test_path = "../Case/BRCASubtype/" + real_input
    if real_input == 'BRCASubtypeSel_test.csv' and not os.path.exists(path=test_path):
        with pkg_resources.open_text('syng_bts.Case.BRCASubtype', 'BRCASubtypeSel_test.csv') as data_file:
            real = pd.read_csv(data_file)
    else:
        real = pd.read_csv(test_path, header = 0)

    # Define the default group level
    level0 = real['groups'].iloc[0]
    level1 = list(set(real['groups']) - set([level0]))

    # Get sample groups
    groups_real = pd.Series(np.where(real['groups'] == "Infiltrating Ductal Carcinoma", "Ductal", "Lobular"))

    groups_generated = pd.Series(np.where(generated.iloc[:, -1] == 1, "Ductal", "Lobular"))

    # Get pure data matrices
    real = real.select_dtypes(include=[np.number])
    real = np.log2(real + 1)
    generated = generated.iloc[:, :real.shape[1]]
    generated.columns = real.columns

    # Select samples for analysis to save running time
    real_ind = list(range(200)) + list(range(len(real) - 200, len(real)))
    generated_ind = list(range(200)) + list(range(len(generated) - 200, len(generated)))

    # Call evaluation functions
    h_subtypes = heatmap_eval(dat_real = real.iloc[real_ind,], dat_generated = generated.iloc[generated_ind,])
    p_umap_subtypes = UMAP_eval(dat_real = real.iloc[real_ind,],
                                dat_generated = generated.iloc[generated_ind,],
                                groups_real = groups_real.iloc[real_ind],
                                groups_generated = groups_generated.iloc[generated_ind],
                                legend_pos = "bottom")

# evaluation()
