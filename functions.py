from itertools import combinations, permutations, product
from collections import Counter
import dateutil.parser
from nltk.corpus import wordnet as wn
import recordlinkage.index
import jellyfish
import pandas as pd
import math


# BEFORE UNIQUE/FOREIGN
def remove_columns_by_category_mismatch(left, right):
    """
    The functions takes two dataframes and removes the columns that cannot be matched, based on their datatypes.
    :param left: Left Dataframe
    :param right: Right Dataframe
    :return: (Updated Left, Updated Right, Discarded columns from Left, Discarded columns from Right)
    """
    lcols_by_cat = get_columns_grouped_by_category(left)
    rcols_by_cat = get_columns_grouped_by_category(right)
    cats = [x for x in lcols_by_cat.keys()]
    [cats.append(x) for x in rcols_by_cat.keys() if x not in cats]
    left_discard = []
    right_discard = []
    for cat in cats:
        if cat not in lcols_by_cat.keys():
            [right_discard.append(col) for col in rcols_by_cat[cat]]
        elif cat not in rcols_by_cat.keys():
            [left_discard.append(col) for col in lcols_by_cat[cat]]
    new_left_cols = [col for col in left.columns if col not in left_discard]
    new_right_cols = [col for col in right.columns if col not in right_discard]
    return left[new_left_cols], right[new_right_cols], left_discard, right_discard


def remove_columns_by_uniqueness(df, min_uniqs=1, min_uniq_ratio=0):
    """
    The functions takes as input a Dataframe and returns the Dataframe without the columns that do not satisfy the uniqueness threshold.
    :param df: Dataframe
    :param min_uniqs: Minimum unique values
    :param min_uniq_ratio: Minimum unique values to total rows (ratio)
    :return: (Sliced Dataframe, discarded columns)
    """
    tot_rows = len(df)
    threshold = max(min_uniqs, min_uniq_ratio * tot_rows)
    new_columns = []
    discarded_columns = []
    if threshold > 0:
        for col in df.columns:
            if df[col].unique() > threshold:
                new_columns.append(col)
            else:
                discarded_columns.append(col)
        return df[new_columns], discarded_columns
    else:
        return df, []


def remove_columns_by_missing_values(df, max_normalized_missing_values=1):
    """
    :param df: Dataframe
    :param max_normalized_missing_values: maximum normalized number of missing values accepted
    :return: a tuple with the new dataframe as the first element and discarded columns as the second one
    """
    tot_rows = len(df)
    missing_values = df.isna().sum()
    new_columns = []
    discarded_columns = []
    for index, value in missing_values.items():
        norm_missing_values = value / tot_rows
        if norm_missing_values < max_normalized_missing_values:
            new_columns.append(index)
        else:
            discarded_columns.append((index, value))
    return df[new_columns], discarded_columns


# BEFORE ANALYSIS
def remove_columns_by_length(df, max_average_characters=50):
    """
    :param df: Dataframe
    :param max_average_characters: maximum number of average characters per column accepted
    :return: (Sliced Dataframe, discarded columns)
    """
    df_dtypes = [(index, value.name) for index, value in df.dtypes.items()]
    new_columns = []
    discarded_columns = []
    for column, datatype in df_dtypes:
        if datatype == 'object':
            avg_chars = df[column].str.len().dropna().agg(["average"])[0]
            if avg_chars < max_average_characters:
                new_columns.append(column)
            else:
                discarded_columns.append((column, avg_chars))
        else:
            new_columns.append(column)
    return df[new_columns], discarded_columns


# BEFORE UNIQUE/FOREIGN
def get_discardable_pairs_by_jaccard(left, right, min_join_score):
    """
    :param left: Left Dataframe
    :param right: Right Dataframe
    :param min_join_score: Minimum Join Score to discard the pairs
    :return: discarded column pairs
    """
    discarded_pairs = []
    if min_join_score > 0:
        lcols = [(col, left[col].unique()) for col in left.columns]
        rcols = [(col, right[col].unique()) for col in right.columns]
        for (lcol, luniq) in lcols:
            for (rcol, runiq) in rcols:
                min = min(luniq, runiq)
                max = max(luniq, runiq)
                if min < max * min_join_score:
                    discarded_pairs.append((lcol, rcol))
    return discarded_pairs


# TO GET COMMON/SIMILAR COLUMNS
def measure_string_similarity(first, second, similarity_function="Equality"):
    """
    The function takes two strings and a similarity function and returns a dict containing a normalized score and the method used
    :param first: first string
    :param second: second string
    :param similarity_function: similarity function used to measure similarity,
        available functions include "Levenshtein", "Hamming", "Semantic" and "Equality" by default
    :return: a dictionary with a normalized score and the similarity method used
    """
    match similarity_function:
        case "Levenshtein":
            score = 1 - jellyfish.levenshtein_distance(first, second) / max(len(first), len(second))
            return {"score": score, "method": "Levenshtein"}
        case "Jaro":
            score = jellyfish.jaro_similarity(first, second)
            return {"score": score, "method": "Jaro"}
        case "Semantic":
            score_first = compute_semantic_similarity(first, first)
            score_second = compute_semantic_similarity(second, second)
            score_absolute = compute_semantic_similarity(first, second)
            score = score_absolute / max(score_first, score_second, 1)
            return {"score": score, "method": "Semantic"}
        case "Equality":
            if first == second:
                score = 1
            else:
                score = 0
            return {"score": score, "method": "Equality"}
        case _:
            if first == second:
                score = 1
            else:
                score = 0
            return {"score": score, "method": "Equality"}


def compute_semantic_similarity(first, second):
    """
    :param first: First String
    :param second: Second String
    :return: normalized similarity score
    """
    synsets_first = wn.synsets(first)
    synsets_second = wn.synsets(second)
    similarities = []
    for s1 in synsets_first:
        for s2 in synsets_second:
            sim = s1.path_similarity(s2)
            if sim is not None:
                similarities.append(sim)
    return max(similarities) if similarities else 0.0


def get_common_columns(left_columns, right_columns, sim_methods=None):
    """
    The function gets as input a list of columns from table1, a list of columns from table2 and a list of similarity methods.
    It returns as output a list of dicts with columnns from the two tables satisfying the similarity methods.
    :param left_columns: list of column names from table1
    :param right_columns: list of column names from table2
    :param sim_methods: list of tuples, with first element as the similarity method and the second one as the minimum score required to satisfy it
    :return: a list of dicts with left column, right column and a list of the similarity methods with their scores
    """
    if sim_methods is None:
        sim_methods = []
    columns = []
    for l_col in left_columns:
        for r_col in right_columns:
            if not sim_methods:
                sim_info = measure_string_similarity(l_col, r_col)
                if sim_info["score"] == 1:
                    columns.append({
                        "left": [l_col],
                        "right": [r_col],
                        "similarity_info": [sim_info]
                    })
            else:
                sim_info_list = [info for (fun, min_score) in sim_methods for info in
                                 [measure_string_similarity(l_col, r_col, fun)] if info["score"] >= min_score]
                if sim_info_list:
                    columns.append({
                        "left": [l_col],
                        "right": [r_col],
                        "similarity_info": sim_info_list
                    })
    return columns


def column_combinations(pair_columns, list_num_elements):
    """
    It takes as input a list of pairs and a list with the number of elements to include in the combinations, and return a list of combinations
    :param pair_columns: List of pairs (each side of the pairs contains one element)
    :param list_num_elements: List of number of elements for the combination
    :return: List of pairs including the combinations
    """
    if len(pair_columns) < min(list_num_elements):
        return []
    res = []
    pairs = [(pair["left"][0], pair["right"][0]) for pair in pair_columns]
    for num_elements in list_num_elements:
        comb_pairs = list(set(combinations(pairs, num_elements)))
        for i in range(0, len(comb_pairs)):
            flag_duplicates = False
            aux = {"left": [], "right": []}
            for x, y in comb_pairs[i]:
                if x not in aux["left"] and y not in aux["right"]:
                    aux["left"].append(x)
                    aux["right"].append(y)
                else:
                    flag_duplicates = True
            if not flag_duplicates:
                res.append(aux)
    return res


# TO GET INSTANCE-MATCHABLE COLUMNS
def get_unique_keys(df, minimum_norm_unique_rows=1, maximum_num_columns=3):
    """
    :param maximum_num_columns: maximum number of columns per unique key
    :param df: Dataframe for which to find the unique keys
    :param minimum_norm_unique_rows: minimum percentage of unique rows required for a column to be part of a key
    :return: list of dicts with the number of columns in the key, the columns and the percentage of unique rows
    """
    i = 0
    num_rows = len(df)
    res = []
    max_cycles = min(maximum_num_columns, len(df.columns))
    while res == [] and i < max_cycles:
        for col_names in combinations(df.columns, i + 1):
            num_unique_rows = len(df[list(col_names)].drop_duplicates())
            if num_unique_rows / num_rows >= minimum_norm_unique_rows:
                res.append({
                    "number": i + 1,  # number of columns
                    # "columns": [{"name": x, "type": df.dtypes[x].name} for x in col_names],
                    'columns': [{'name': x, 'category': get_column_category(df[x])} for x in col_names],
                    "percentage": num_unique_rows / num_rows
                })
        i += 1

    return res


def find_foreign_keys(left_prim_keys, right, keys_table="left"):
    """
    Given the unique keys of one table, the function returns the matching keys in the other table, based on the category
    :param left_prim_keys: Primary (or unique keys)
    :param right: The other Dataframe
    :param keys_table: A string to identify the keys' table
    :return: List of matching pairs
    """
    r_cols_by_cat = get_columns_grouped_by_category(right)
    pairs = []
    for p_key in left_prim_keys:
        p_cols_by_cat = {}
        for elem in p_key["columns"]:
            p_cols_by_cat[elem['category']] = [elem['name']] if elem['category'] not in p_cols_by_cat.keys() else \
                p_cols_by_cat[elem['category']] + [elem['name']]

        if all(cat in r_cols_by_cat.keys() for cat in p_cols_by_cat.keys()):
            matches = []
            for col in p_key['columns']:
                if not matches:
                    matches = r_cols_by_cat[col['category']]
                else:
                    aux_matches = []
                    for old_pair in matches:
                        for new_part in r_cols_by_cat[col['category']]:
                            if new_part not in old_pair:
                                aux_matches.append([old_pair] + [new_part])
                    matches = aux_matches
            p_col_names = [col['name'] for col in p_key['columns']]
            for match in matches:
                match = match if isinstance(match, list) else [match]
                if keys_table == "left":
                    pairs.append((p_col_names, match))
                else:
                    pairs.append((match, p_col_names))
    return pairs


# TO MEASURE JOINABILITY
def measure_join_columns(left, right, left_columns, right_columns, minimum_join_score, method="SortedNeighborhood",
                         score_method="jaccard"):
    """
    :param minimum_join_score:
    :param left: Left Dataframe
    :param right: Right Dataframe
    :param left_columns: list of left column names
    :param right_columns: list of right column names
    :param method: can be either "Merge" with the use of Dataframe functions or "SortedNeighborhood"
    :return: returns a normalized score between 0 and 1, as only the unique values are checked
    """
    lc_list = left_columns if isinstance(left_columns, list) else [left_columns]
    rc_list = right_columns if isinstance(right_columns, list) else [right_columns]

    same_type_flag = True
    counter = 0

    if len(lc_list) != len(rc_list):
        return 0, 0, 0, "error - length mismatch"
    else:
        if not (set(lc_list).issubset(set(left.columns)) and set(rc_list).issubset(set(right.columns))):
            return 0, 0, 0, "error - wrong column names"
        while same_type_flag and counter < len(lc_list):
            # if left.dtypes[lc_list[counter]] != right.dtypes[rc_list[counter]]:
            cat_left = get_column_category(left[lc_list[counter]])
            cat_right = get_column_category(right[rc_list[counter]])
            if cat_left != cat_right:
                same_type_flag = False
                message = 'Category mismatch: ' + cat_left + '-' + cat_right
            counter += 1
        if not same_type_flag:
            return 0, 0, 0, message
        else:
            sliced_left = left[lc_list].drop_duplicates()
            sliced_right = right[rc_list].drop_duplicates()
            len_left = len(sliced_left)
            len_right = len(sliced_right)
            if score_method == 'jaccard':
                max_score_achievable = min(len_left, len_right) / max(len_left, len_right)
            elif score_method == 'cosine':
                max_score_achievable = min(len_left, len_right) / math.sqrt(len_left * len_right)
            else:
                max_score_achievable = 1
            if max_score_achievable < minimum_join_score:
                return 0, 0, 0, "Pruned SN (" + str("{:.2f}".format(max_score_achievable)) + ")"
            if method == "Merge":
                merged = sliced_left.merge(sliced_right, left_on=lc_list, right_on=rc_list, how="inner")
                common_uniq_vals = len(merged)
                tot_uniq_vals = len_left + len_right - common_uniq_vals
                return common_uniq_vals / tot_uniq_vals, common_uniq_vals, tot_uniq_vals
            else:
                indexer = recordlinkage.index.SortedNeighbourhood(left_on=lc_list[0], right_on=rc_list[0],
                                                                  block_left_on=lc_list[1:], block_right_on=rc_list[1:],
                                                                  window=9)
                possible_matches = indexer.index(sliced_left, sliced_right)
                compare = recordlinkage.Compare()
                [compare.exact(lc_list[i], rc_list[i], label=lc_list[i]) for i in range(0, len(lc_list))]
                features = compare.compute(possible_matches, sliced_left, sliced_right)
                matches = features[features.sum(axis=1) > len(lc_list) - 1]
                common_uniq_vals = len(matches)
                if score_method == 'jaccard':
                    tot_uniq_vals = len_left + len_right - common_uniq_vals
                else:
                    tot_uniq_vals = math.sqrt(len_left * len_right)
                return common_uniq_vals / tot_uniq_vals, common_uniq_vals, tot_uniq_vals, "Sorted Neighbourhood"


def get_useful_columns(left_df, right_df, list_of_join_columns_dict, minimum_join_score, evaluation_method,
                       score_method='jaccard'):
    """
    :param left_df: Dataframe of table1
    :param right_df: Dataframe of table2
    :param list_of_join_columns_dict: list of dicts containing list of table1 columns and a list of table2 columns
    :param minimum_join_score: minimum join score required for the join to be considered useful
    :param evaluation_method: parameter to be passed to the evaluation function
    :return: tuple of two lists, first one of useful column pairs and second of pairs to be discarded
    """
    merge = []
    discard = []
    for columns_dict in list_of_join_columns_dict:
        score, matches, total, description = measure_join_columns(left_df, right_df, columns_dict["left"],
                                                                  columns_dict["right"], minimum_join_score,
                                                                  evaluation_method, score_method)
        if score >= minimum_join_score:
            merge.append({
                "left": columns_dict["left"],
                "right": columns_dict["right"],
                "similarity_info": columns_dict["similarity_info"] if columns_dict.get("similarity_info") else [],
                "join_score": score,
                "matches": matches,
                "total": total,
                "description": description
            })
        else:
            discard.append({
                "left": columns_dict["left"],
                "right": columns_dict["right"],
                "similarity_info": columns_dict["similarity_info"] if columns_dict.get("similarity_info") else [],
                "join_score": score,
                "matches": matches,
                "total": total,
                "description": description,
            })
    return merge, discard


def measure_join(left, right, left_join_columns, right_join_columns):
    """
    Measure the join obtained between the two tables as the contribution obtained by the other
    :param left: Left Dataframe
    :param right: Right Dataframe
    :param left_join_columns: Left columns
    :param right_join_columns: Right columns
    :return: a ratio of new values to existing values
    """
    joined_df = left.merge(right, left_on=left_join_columns, right_on=right_join_columns, how="inner",
                           suffixes=('_left', '_right'))

    r_j_cols_merged = [r_col + "_right" if r_col in left.columns else r_col for r_col in right_join_columns]
    r_cols_merged = [r_col + "_right" if r_col in left.columns else r_col for r_col in right.columns]
    # Contribution of the Dataframe right to the Dataframe left
    contribution_r_df = joined_df[[col for col in r_cols_merged if col not in r_j_cols_merged]].dropna(how='all')
    contr_r = (len(r_cols_merged) - len(r_j_cols_merged)) * len(contribution_r_df.index)
    contr_r_perc = contr_r * 100 / (len(left.index) * len(left.columns))

    l_j_cols_merged = [l_col + "_left" if l_col in right.columns else l_col for l_col in left_join_columns]
    l_cols_merged = [l_col + "_left" if l_col in right.columns else l_col for l_col in left.columns]
    # Contribution of the Dataframe left to the Dataframe right
    contribution_l_df = joined_df[[col for col in l_cols_merged if col not in l_j_cols_merged]].dropna(how='all')
    contr_l = (len(l_cols_merged) - len(l_j_cols_merged)) * len(contribution_l_df.index)
    contr_l_perc = contr_l * 100 / (len(right.index) * len(right.columns))

    return contr_r_perc, contr_l_perc


# AUXILIARY FUNCTIONS
def col_pair_in(left_columns, right_columns, pairs_list):
    """
    The function is used to check if a pair of columns, left and right, are contained in a list of pairs.
    :param left_columns: Left columns
    :param right_columns: Right columns
    :param pairs_list: List of pairs
    :return: Boolean value, True if the pair is in the list, false if it's not
    """
    lc_list = left_columns if isinstance(left_columns, list) else [left_columns]
    rc_list = right_columns if isinstance(right_columns, list) else [right_columns]
    if len(lc_list) != len(rc_list):
        return False
    same_length_pairs = list(filter(lambda x: len(x["left"]) == len(lc_list), pairs_list))
    for pair in same_length_pairs:
        lc_list_in_pair = pair["left"] if isinstance(pair["left"], list) else [pair["left"]]
        rc_list_in_pair = pair["right"] if isinstance(pair["right"], list) else [pair["right"]]
        lc_list_in_pair.sort()
        lc_list.sort()
        rc_list_in_pair.sort()
        rc_list.sort()
        if lc_list_in_pair == lc_list and rc_list_in_pair == rc_list:
            return True
    return False


def col_pair_is_child_in(left_columns, right_columns, pairs_list, return_list="False"):
    """
    :param left_columns: List of Left columns
    :param right_columns: List of Right columns
    :param pairs_list: List of Pairs
    :param return_list: True or False, depending on whether the list of pairs should be returned or not
    :return: either a Boolean value or (Boolean, Lists), depending on the value of "return_list"
    """
    lc_list = left_columns if isinstance(left_columns, list) else [left_columns]
    rc_list = right_columns if isinstance(right_columns, list) else [right_columns]
    if len(lc_list) != len(rc_list):
        if return_list:
            return False, (left_columns, right_columns)
        else:
            return False
    child_list_of_pairs = lists_to_list_of_pairs(lc_list, rc_list)
    all_parent_pairs = list(filter(lambda x: len(x["left"]) < len(lc_list), pairs_list))
    for parent_pair_list in all_parent_pairs:
        parent_list_pairs = lists_to_list_of_pairs(parent_pair_list["left"], parent_pair_list["right"])
        for parent_pair in parent_list_pairs:
            if parent_pair in child_list_of_pairs:
                child_list_of_pairs = list(filter(lambda x: x != parent_pair, child_list_of_pairs))
                flagChanged = True
    if return_list:
        return flagChanged, list_of_pairs_to_lists(child_list_of_pairs)
    else:
        return flagChanged


def lists_to_list_of_pairs(left, right):
    """
    Convert two lists of columns, left and right, to a list of pairs
    :param left: List of left columns
    :param right: list of right columns
    :return: List of (left column, right column)
    """
    if len(left) != len(right):
        return []
    list_of_pairs = []
    i = 0
    for i in range(0, len(left)):
        list_of_pairs.append((left[i], right[i]))
    return list_of_pairs


def list_of_pairs_to_lists(list_pairs):
    """
    Convert a list of pairs to a list of left columns and a list of right columns
    :param list_pairs: List of pairs
    :return: (List of left columns, List of right columns)
    """
    left = []
    right = []
    for x, y in list_pairs:
        left.append(x)
        right.append(y)
    return left, right


def show_columns(columns):
    """
    The function is used to print the columns (list of dicts)
    :param columns: list of dicts containing the left columns,
    :return:
    """
    for column in columns:
        if 'description' in column:
            description = column['description']
        else:
            description = ""
        print("\tLeft:", ", ".join(column['left']),
              "\tRight:", ", ".join(column['right']),
              "\tJoin score:", "{:.2f}".format(100 * column['join_score']),
              "(", column['matches'], "/", column['total'], ")", description)
    return


def get_columns_grouped_by_category(df):
    """
    Group columns by category
    :param df: Dataframe
    :return: Dict of {category: List of column names}
    """
    cats = {}
    for col_name in df.columns:
        cat = get_column_category(df[col_name])
        if cat in cats.keys():
            cats[cat].append(col_name)
        else:
            cats[cat] = [col_name]
    return cats


def get_column_category(series):
    """
    Given a series, it returns the category the series belongs to
    :param series: Series
    :return: category (i.e. Boolean, Numerical, Date or Text)
    """
    if pd.api.types.is_bool_dtype(series):
        return 'Boolean'
    elif pd.api.types.is_numeric_dtype(series):
        return 'Numerical'
    else:
        s_df = pd.to_datetime(series, infer_datetime_format=True, errors='coerce', dayfirst=True)
        s_mf = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
        length = len(s_df)
        date_vals = max(s_df.apply(lambda x: x == x).sum(), s_mf.apply(lambda x: x == x).sum())
        if date_vals / length > 0.5:
            return 'Date'
        else:
            return 'Text'


def check_if_column_counter(dataframe, column_name, min_uniqueness=0.99, max_miss_incr_values=0.1):
    """
    Check if a column is a counter
    :param dataframe: Dataframe
    :param column_name: Column name
    :param min_uniqueness: Minimum uniqueness per counter column
    :param max_miss_incr_values: Maximum missing incremental values per counter column
    :return:
    """
    col_dtype = dataframe.dtypes[column_name].name
    if col_dtype == "int64" or col_dtype == "float64":
        total_rows = len(dataframe[column_name])
        unique_rows = dataframe[column_name].nunique()
        norm_unique_rows = unique_rows / total_rows
        if norm_unique_rows < min_uniqueness:
            return False
        dataframe.sort_values(column_name, ascending=True, inplace=True)
        diff_list = dataframe[column_name].diff().dropna().tolist()
        num_most_freq_elem = count_most_frequent(diff_list)
        miss_incr = 1 - num_most_freq_elem / len(diff_list)
        if miss_incr <= max_miss_incr_values:
            return True
    return False

def count_most_frequent(lst):
    """
    Count the occurrences of the most frequent element in the list
    :param lst: List of values
    :return: Number of occurrences of the most frequent element in lst
    """
    if not lst:
        return 0
    count = Counter(lst)
    return max(count.values())
