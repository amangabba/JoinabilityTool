# LIBRARIES
import os
import pandas as pd
import functions as f




def main():
    # (method_name, minimum_limit)
    list_sim_method = [("Levenshtein", 0.5), ("Jaro", 0.8), ("Semantic", 0.5)]
    min_join = 0.45  # minimum accepted score for a join
    min_norm_uniq_rows = .95  # minimum percentage of unique rows
    max_norm_missing_values = 0.3
    max_avg_column_chars = 70
    thres_hier_pair_discard = 1
    eval_method = "SortedNeighborhood"
    # eval_method = "Merge"
    score_method = 'jaccard'
    # score_method = 'cosine'

    absolute_path = os.path.dirname(__file__)
    source_path = os.path.join(absolute_path, "./data/LibraryBooks.csv")
    target_path = os.path.join(absolute_path, "./data/BookReviews.csv")
    source_original = pd.read_csv(source_path)
    target_original = pd.read_csv(target_path)



    print("Table 1 has", len(source_original), "rows and", len(source_original.columns), "columns!")
    print("Table 2 has", len(target_original), "rows and", len(target_original.columns), 'columns!\n')


    print("Removing columns by missing values...")
    source, removed_columns = f.remove_columns_by_missing_values(source_original, max_norm_missing_values)
    [print("Removed column", col_name, "from \"Table 1\", it has", miss_values, "missing values.",) for
     col_name, miss_values in removed_columns]
    target, removed_columns = f.remove_columns_by_missing_values(target_original, max_norm_missing_values)
    [print("Removed column", col_name, "from \"Table 2\", it has", miss_values, "missing values.", ) for
     col_name, miss_values in removed_columns]

    print("\nRemoving columns by average number of characters...")
    source, removed_columns = f.remove_columns_by_length(source, max_avg_column_chars)
    [print("Removed column", col_name, "from \"Table 1\", it has", avg_chars, "average characters.", ) for
     col_name, avg_chars in removed_columns]
    target, removed_columns = f.remove_columns_by_length(target, max_avg_column_chars)
    [print("Removed column", col_name, "from \"Table 2\", it has", avg_chars, "average characters.", ) for
     col_name, avg_chars in removed_columns]

    print("\nTable 1 has", len(source), "rows and", len(source.columns), "columns!")
    print("Table 2 has", len(target), "rows and", len(target.columns), 'columns!\n')

    # COMMON COLUMNS
    cc_sing = f.get_common_columns(source.columns, target.columns)
    cc_sing_merge, cc_sing_disc = f.get_useful_columns(source, target, cc_sing, min_join, eval_method, score_method)
    discard_hier = [x for x in cc_sing_disc if x['matches'] <= thres_hier_pair_discard]

    cc_multi = []
    [cc_multi.append(pair) for pair in f.column_combinations(cc_sing, [2, 3]) if not f.col_pair_is_child_in(pair['left'], pair['right'], discard_hier)]
    cc_multi_merge, cc_multi_dis = f.get_useful_columns(source, target, cc_multi, min_join, eval_method, score_method)

    merge_columns = cc_sing_merge + cc_multi_merge
    cc_disc = cc_sing_disc + cc_multi_dis
    print("\nMergeable (COMMON COLUMNS):")
    f.show_columns(merge_columns)
    print("Discarded (COMMON COLUMNS):")
    f.show_columns(cc_disc)

    discarded = cc_disc
    merge_columns = sorted(merge_columns, key=lambda x: x["join_score"], reverse=True)
    if not merge_columns:
        # SIMILAR COLUMNS
        sc_sing = f.get_common_columns(source.columns, target.columns, list_sim_method)
        sc_sing = [col for col in sc_sing if not f.col_pair_in(col['left'], col['right'], discarded)]
        sc_sing_merge, sc_sing_disc = f.get_useful_columns(source, target, sc_sing, min_join, eval_method, score_method)
        [discard_hier.append(x) for x in sc_sing_disc if x['matches'] <= thres_hier_pair_discard] # UPDATE HIER. DISCARD

        [discarded.append(col) for col in sc_sing_disc if col not in discarded] # UPDATE DISCARD

        sc_multi = []
        for pair in f.column_combinations(sc_sing + cc_sing, [2, 3]):
            if not f.col_pair_in(pair['left'], pair['right'], discarded):
                if not f.col_pair_is_child_in(pair['left'], pair['right'], discard_hier):
                    sc_multi.append(pair)
        sc_multi_merge, sc_multi_disc = f.get_useful_columns(source, target, sc_multi, min_join, eval_method, score_method)

        sc_merge = sc_sing_merge + sc_multi_merge
        sc_disc = sc_sing_disc + sc_multi_disc

        merge_columns = sc_merge
        [discarded.append(col) for col in sc_multi_disc if col not in discarded] # UPDATE DISCARD

        print("\nMergeable (SIMILAR COLUMNS):")
        f.show_columns(merge_columns)
        print("Discarded (AFTER SIMILAR COLUMNS SEARCH):")
        f.show_columns(sc_disc)
        if not merge_columns:
            # NO COMMON/SIMILAR COLUMNS
            source, target, s_disc, t_disc = f.remove_columns_by_category_mismatch(source, target)
            print("Discarded columns for category mismatch: ", s_disc, t_disc)
            im_columns = []
            source_keys = f.get_unique_keys(source, min_norm_uniq_rows)
            target_keys = f.get_unique_keys(target, min_norm_uniq_rows)

            if (not source_keys) and (not target_keys):
                print("\nThe two tables do not seem to have any valid unique key")
            else:
                if source_keys:
                    [im_columns.append({"left": x, "right": y, "similarity_info": []}) for x, y in
                     f.find_foreign_keys(source_keys, target)]
                if target_keys:
                    [im_columns.append({"left": x, "right": y, "similarity_info": []}) for x, y in
                     f.find_foreign_keys(target_keys, source, keys_table="right")]
                im_columns = [col for col in im_columns if not f.col_pair_in(col["left"], col["right"], discarded)]
                for original, list_to_append in zip((merge_columns, discarded),
                                                    f.get_useful_columns(source, target, im_columns, min_join, eval_method, score_method)):
                    [original.append(col) for col in list_to_append]

                print("\nMergeable (PRIMARY/FOREIGN JOIN COLUMNS):")
                f.show_columns(merge_columns)
                print("Discarded (PRIMARY/FOREIGN JOIN COLUMNS):")
                f.show_columns(discarded)

            if not merge_columns:
                cm_pairs_found = []
                cm_pairs = []
                s_cols_by_cat = f.get_columns_grouped_by_category(source)
                t_cols_by_cat = f.get_columns_grouped_by_category(target)
                cats = s_cols_by_cat.keys()
                [cats.append(x) for x in t_cols_by_cat if x not in cats]
                for cat in s_cols_by_cat:
                    if cat in t_cols_by_cat.keys():
                        for l_col in s_cols_by_cat.get(cat):
                            for r_col in t_cols_by_cat.get(cat):
                                cm_pairs_found.append({'left': [l_col], 'right': [r_col], 'similarity_info': []})
                [cm_pairs.append(x) for x in cm_pairs_found if not f.col_pair_in(x['left'], x['right'], discarded)]
                merge_columns, cm_discard = f.get_useful_columns(source, target, cm_pairs, min_join, eval_method, score_method)
                print("\nMergeable (COLUMN MATCHING COLUMNS):")
                f.show_columns(merge_columns)
                print("Discarded (COLUMN MATCHING COLUMNS):")
                f.show_columns(cm_discard)

    if merge_columns:
        print("\nThe two tables can be joined together through the following columns:")
        f.show_columns(merge_columns)
        useful_cols = {"left": [], "right": []}
        for column_pair in merge_columns:
            [useful_cols["left"].append(l_col) for l_col in column_pair["left"] if l_col not in useful_cols["left"]]
            [useful_cols["right"].append(r_col) for r_col in column_pair["right"] if r_col not in useful_cols["right"]]
        counter_cols = {"left": [], "right": []}
        [counter_cols["left"].append(col) for col in useful_cols["left"] if f.check_if_column_counter(source, col)]
        [counter_cols["right"].append(col) for col in useful_cols["right"] if f.check_if_column_counter(target, col)]
        if counter_cols["left"] or counter_cols["right"]:
            print("\nHowever some columns selected for the join from the two tables may be just counters that happen to match!")
            if counter_cols["left"]:
                print("Counter columns in LEFT: ", counter_cols["left"])
            if counter_cols["right"]:
                print("Counter columns in RIGHT: ", counter_cols["right"])

        for pair in merge_columns:
            l_add, r_add = f.measure_join(source_original, target_original, pair["left"], pair["right"])
            print("The right table adds", "{:.2f}".format(l_add), "% more information to the left one, and", "{:.2f}".format(r_add), "% vice versa.")
    else:
        print("\nThe two tables cannot be joined together.")


if __name__ == '__main__':
    main()
