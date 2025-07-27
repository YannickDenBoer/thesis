from preprocessing import prepare_data
from cross_validation import grouped_iterative_multilabel_stratified_cross_validation
from utils import load_data, save_data, sanitize_and_report, CUE_LIST, make_latex_table
from baselines import random_classifier, majority_classifier, weighted_classifier
from evaluate import evaluate, summarize_labelwise
from Models.model_ensemble import Model, SentenceModel, Processor
from thresholds import find_local_thresholds, find_global_threshold, convert_scores_to_labels
import pandas as pd

if __name__ == "__main__":
    inference = True
    baselines = True
    
    # Paths
    video_folder = ""
    annotation_file = ""
    results_folder = ""
    
    # Initialise model and processor/wrapper
    vl_model = Model()
    sentence_model = SentenceModel()
    processor = Processor(vl_model, sentence_model)
    
    # Prepare Data
    folder = "Data/Results/Main/data.csv"
    if inference:
        X, y, groups = prepare_data(video_folder, annotation_file)
        save_data((X,y,groups), folder)
    else:
        X, y, groups = load_data(folder)

    # Calculate Embedding voor all X
    true_answer_embeddings = processor.true_answers_to_embeddings()
    print(f"true_answer_embeddings: {true_answer_embeddings.shape}")
    if inference:
        X = processor.images_to_embeddings(X)
        save_data(X, results_folder + "embeddings.pkl")
    else:
        X = load_data(results_folder + "embeddings.pkl")

    # Clean NaN values
    X, y, groups = sanitize_and_report(X, "train_cos_scores", y_true=y, label_names=CUE_LIST, group = groups)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Calculate Cos Similarity scores
    cos_scores = processor.embeddings_to_cos_scores(X, true_answer_embeddings)
    print(f"cos_scores: {cos_scores.shape}")

    # Initialise results lists
    results_baseline_random = []
    results_baseline_majority = []
    results_baseline_weighted = []
    results_globalt = []
    results_localt = []
    results_yesnocosim = []

    # Perform k-fold cross-validation and split the data into train and test sets
    splits = grouped_iterative_multilabel_stratified_cross_validation(X, y, groups, n_splits=5, epsilon=2, random_state=123)

    # For each fold
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/{len(splits)}")
        folder = f"{results_folder}fold_{fold + 1}/"

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        cos_scores_train, cos_scores_test = cos_scores[train_idx], cos_scores[test_idx]

        num_test_samples = len(y_test)
        num_test_labels = len(y_test[0])

        ### TRAIN ###
        # Baselines
        y_pred_random = random_classifier(num_test_samples, num_test_labels)
        y_pred_majority = majority_classifier(y_train, num_test_samples, num_test_labels)
        y_pred_weighted = weighted_classifier(y_train, num_test_samples, num_test_labels)

        # Calculate thresholds
        global_thresholds, f1_g = find_global_threshold(y_train, cos_scores_train)
        local_thresholds, f1_l = find_local_thresholds(y_train, cos_scores_train)

        print(local_thresholds, global_thresholds)
        #TODO: Yes/no thresholds
        #TODO: Mean/std threshold(s)

        y_pred_global = convert_scores_to_labels(cos_scores_test, global_thresholds)
        y_pred_local = convert_scores_to_labels(cos_scores_test, local_thresholds)

        ### EVAL ###
        if baselines:
            results_baseline_random.append(evaluate(y_test, y_pred_random, label_names=CUE_LIST))
            results_baseline_majority.append(evaluate(y_test, y_pred_majority, label_names=CUE_LIST))
            results_baseline_weighted.append(evaluate(y_test, y_pred_weighted, label_names=CUE_LIST))
        
        #print(f"Global Threshold Classifier:")
        results_globalt.append(evaluate(y_test, y_pred_global))
        #print(f"Local Threshold Classifier:")
        results_localt.append(evaluate(y_test, y_pred_local))

    summary_keys = [
        "micro_precision","macro_precision",
        "micro_recall",   "macro_recall",
        "micro_f1",       "macro_f1",
        "subset_accuracy","hamming_loss",
        "jaccard_micro",  "jaccard_macro"
    ]
    #baselines (you only have to do this once)
    if baselines:
        
        df_rand = pd.DataFrame(results_baseline_random)[summary_keys]
        df_maj  = pd.DataFrame(results_baseline_majority)[summary_keys]
        df_wtd  = pd.DataFrame(results_baseline_weighted)[summary_keys]

        # 1) Compute mean/std and transpose so metrics become the index
        summary_random   = df_rand.agg(['mean','std']).T
        summary_majority = df_maj.agg(['mean','std']).T
        summary_weighted = df_wtd.agg(['mean','std']).T

        # 2) Give each DataFrame a two-level column index: (Baseline, Statistic)
        summary_random.columns   = pd.MultiIndex.from_product([['Random'],   ['mean','std']])
        summary_majority.columns = pd.MultiIndex.from_product([['Majority'], ['mean','std']])
        summary_weighted.columns = pd.MultiIndex.from_product([['Weighted'], ['mean','std']])

        # 3) Concatenate them side by side
        summary_all = pd.concat(
            [summary_random, summary_majority, summary_weighted],
            axis=1
        )
        print("Baseline summary:")
        print(summary_all)
        print("Baseline perlabel random")
        perlabel_random = summarize_labelwise(results_baseline_random, CUE_LIST)
        print(perlabel_random)
        print("Baseline perlabel majority")
        perlabel_majority = summarize_labelwise(results_baseline_majority, CUE_LIST)
        print(perlabel_majority)
        print("Baseline perlabel weighted")
        perlabel_weighted = summarize_labelwise(results_baseline_weighted, CUE_LIST)
        print(perlabel_weighted)
        #print(make_latex_table(summary_all, "Baseline Summary", "tab:baselines"))
        print(make_latex_table(perlabel_random, "Baseline Random", "tab:baseline_random"))
        print(make_latex_table(perlabel_majority, "Baseline Majority", "tab:baseline_majority"))
        print(make_latex_table(perlabel_weighted, "Baseline Weighted", "tab:baseline_weighted"))

    print(f"Model Global/local summary:")
    df_global = pd.DataFrame(results_globalt)[summary_keys]
    df_local = pd.DataFrame(results_localt)[summary_keys]
    summary_global = df_global.agg(['mean','std']).T
    summary_local = df_local.agg(['mean','std']).T

    summary_global.columns = pd.MultiIndex.from_product([['Global'], ['mean','std']])
    summary_local.columns = pd.MultiIndex.from_product([['Local'], ['mean','std']])

    summary_all = pd.concat([summary_global, summary_local], axis=1)
    print(summary_all)
    print(f"Global model per label summary:")
    perlabel_global = summarize_labelwise(results_globalt, CUE_LIST)
    print(f"Local model per label summary:")
    perlabel_local = summarize_labelwise(results_localt, CUE_LIST)

    save_data(summary_all, results_folder + "summary.pkl")
    save_data(perlabel_global, results_folder + "perlabel_global.pkl")
    save_data(perlabel_local, results_folder + "perlabel_local.pkl")

    print(make_latex_table(perlabel_local, "Qwen Qwen Answer Only Local", "tab:qwenqwenanswer_local", float_fmt="%.2f"))
    print(make_latex_table(perlabel_global, "Qwen Qwen Answer Only Global", "tab:qwenqwenanswer_global", float_fmt="%.2f"))
    print(make_latex_table(summary_all, "Qwen Qwen Answer Only Summary", "tab:qwenqwenanswer_summary", float_fmt="%.2f"))
