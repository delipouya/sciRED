from .corr import(
    get_factor_libsize_correlation
)

from .preprocess import(
    utils_test,
    get_data_array,
    get_highly_variable_gene_indices,
    get_sub_data,
    get_binary_covariate_v1,
    get_binary_covariate,
    get_design_mat,
    get_library_design_mat,
    get_scaled_vector,
)

from .visualize import(
    plot_pca,
    plot_factor_scatter,
    plot_factor_loading,
    plot_umap,
    plot_FCAT,
    plot_histogram,
    plot_matched_factor_dist,
    plot_matched_covariate_dist,
    plot_factor_cor_barplot,
    plot_FIST,
    plot_sorted_factor_FCA_scores,
    plot_relativeVar,
)

from .ex_visualize import(
    get_legend_patch,
    get_colors_dict_scMix,
    get_colors_dict_ratLiver,
    get_colors_dict_humanLiver,
    get_colors_dict_humanKidney,
    get_colors_dict_humanPBMC
)

from .ex_preprocess import(
    import_AnnData,
    get_metadata_scMix,
    get_metadata_ratLiver,
    get_metadata_humanLiver,
    get_metadata_humanKidney,
    get_metadata_humanPBMC,
)

from .simulation import(
    simulate_gaussian,
    simulate_mixture_gaussian,
    calc_overlap_double_Gaussian,
    get_random_number,
    get_random_list,
    get_random_list_sum_1,
    get_random_factor_parameters,
    get_a_factor_pairwise_overlap,
    get_simulated_factor_object,
    get_sim_factor_covariates,
    get_covariate_freq_table,
    get_pairwise_match_score_matrix,
    convert_matrix_list_to_vector,
    mask_upper_triangle,
    plot_scatter,
    get_arithmatic_mean_df,
    get_geometric_mean_df,
)

#__all__ = ['corr', 'preprocess', 'visualize', 'ex_visualize', 'ex_preprocess', 'simulation']