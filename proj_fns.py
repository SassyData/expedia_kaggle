import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Useful Fns for notebooks

def we_trip(row):
    """Returns bool to indicate whether this is a weekend trip, based on check in, check out & length of stay"""
    we_start = 1 if (row.srch_ci_day > 3) and (row.srch_ci_day < 7) else 0
    we_end = 1 if (row.srch_co_day > 5) or (row.srch_co_day < 2) else 0
    short_stay = 1 if (row.srch_los < 6) else 0
    return 1 if (we_start + we_end + short_stay == 3) else 0


def extract_date_info(dt_in, dt_out, dt_format='%Y-%m-%d %H:%M:%S'):
    """Extract info about dates and times"""
    time = datetime.strptime(dt_in, dt_format).time()
    date = datetime.strptime(dt_in, dt_format).date()
    if dt_out == 'hour':
        return time.hour
    elif dt_out == 'month':
        return date.month
    elif dt_out == 'year':
        return date.year


def oh_location(train_raw, orig_col, prefix, keep_loc):
    """One hot encode location info, keeping only locations as specified
    (generally those with more than 1000 searches)"""
    train_raw[orig_col] = train_raw[orig_col].astype(str)
    one_hot_df = pd.get_dummies(train_raw[orig_col])
    # Add prefix to keep locations
    keep_name = [prefix + c.replace(" ", "_") for c in keep_loc]
    # OH encode & remove dups
    one_hot_df.columns = [prefix + name.replace(" ", "_") for name in one_hot_df.columns]
    train_raw = train_raw.join(one_hot_df[keep_name]).drop([orig_col], axis=1)

    return train_raw


def preprocess_step(train_raw):
    """
    Takes Raw df, removes undeeded cols, creates new features.
    Returns: preprocessed df (not yet on-hot encoded)
    """
    # Split out Search Booking Window into weeks
    bins = [i for i in range(0, 364, 7)] + [456, 547]  # bins weekly for 1 yr, then 3 monthly
    train_raw["srch_bw_weeks"] = pd.cut(train_raw.srch_bw, bins, labels=[i for i in range(1, len(bins), 1)])
    train_raw["srch_bw_weeks"] = train_raw["srch_bw_weeks"].cat.add_categories(0)
    train_raw["srch_bw_weeks"].fillna(0, inplace=True)
    train_raw["srch_bw_weeks"] = train_raw["srch_bw_weeks"].astype('int')

    # Split out prop_price_without_discount_usd into bands
    bins = [i for i in range(0, 1050, 150)] + [i for i in range(2000, 6000, 1000)] + \
           [i for i in range(6000, 10000, 2000)] + [100000, 2000000]
    train_raw["prop_price_without_discount_usd_bins"] = pd.cut(train_raw.prop_price_without_discount_usd,
                                                               bins,
                                                               labels=[i for i in range(1, len(bins), 1)])
    train_raw["prop_price_without_discount_usd_bins"] = train_raw[
        "prop_price_without_discount_usd_bins"].cat.add_categories(0)
    train_raw["prop_price_without_discount_usd_bins"].fillna(0, inplace=True)
    train_raw["prop_price_without_discount_usd_bins"] = train_raw["prop_price_without_discount_usd_bins"].astype('int')

    # Calculate Property discount
    train_raw['prop_discount'] = train_raw.apply(lambda x: round(x.prop_price_with_discount_usd /
                                                                 (x.prop_price_without_discount_usd + 0.01), 2) * 100,
                                                 axis=1)

    # One hot encore search Country
    countries_keep = ['UNITED STATES OF AMERICA', 'UNITED KINGDOM', 'SWEDEN', 'FRANCE', 'JAPAN',
                      'NORWAY', 'SOUTH KOREA', 'CANADA', 'HONG KONG', 'BRAZIL', 'TWN', 'DENMARK']
    train_raw = oh_location(train_raw, orig_col='srch_visitor_loc_country', prefix='srch_co_', keep_loc=countries_keep)

    # One hot encore search city
    keep_cit = ['NEW YORK', 'SEOUL', 'LOS ANGELES', 'HONG KONG', 'LAS VEGAS', 'CHICAGO',
                'STOCKHOLM', 'TOKYO', 'WASHINGTON', 'SAN FRANCISCO']
    train_raw = oh_location(train_raw, orig_col='srch_visitor_loc_city', prefix='srch_city_', keep_loc=keep_cit)

    # One hot encore search continent
    keep_cont = ['EUROPE', 'ASIA', 'LATAM']
    train_raw = oh_location(train_raw, orig_col='srch_posa_continent', prefix='srch_cont_', keep_loc=keep_cont)

    # One Hot encode device type
    one_hot_device = pd.get_dummies(train_raw['srch_device'])
    train_raw = train_raw.join(one_hot_device).drop(['WEB'], axis=1)

    # Weekend trip feature (bool)
    train_raw['srch_we_trip'] = train_raw.apply(lambda row: we_trip(row), axis=1)

    # Extract search month & year into new cols
    train_raw['ci_month'] = train_raw.srch_ci.apply(lambda x: extract_date_info(x, dt_out='month',
                                                                                dt_format='%Y-%m-%d'))
    train_raw['ci_year'] = train_raw.srch_ci.apply(lambda x: extract_date_info(x, dt_out='year',
                                                                               dt_format='%Y-%m-%d'))

    # Extract date info
    train_raw['srch_year'] = train_raw.srch_date_time.apply(lambda x: extract_date_info(x, dt_out='year'))
    train_raw['srch_month'] = train_raw.srch_date_time.apply(lambda x: extract_date_info(x, dt_out='month'))
    train_raw['srch_hour'] = train_raw.srch_date_time.apply(lambda x: extract_date_info(x, dt_out='hour'))

    # Cols to drop *AFTER* other preprocessing steps
    drop_cols = ['srch_id', 'srch_visitor_wr_member', 'srch_co', 'srch_ci', 'srch_bw', 'srch_currency',
                 'prop_price_without_discount_local', 'prop_price_with_discount_local', 'srch_visitor_id',
                 'srch_posa_country', 'srch_date_time', 'srch_dest_longitude', 'srch_dest_latitude',
                 'srch_visitor_loc_region', 'srch_device', 'prop_price_without_discount_usd', 'prop_super_region',
                 'prop_continent', 'prop_country', 'srch_local_date']

    train_pre = train_raw.drop(drop_cols, axis=1)
    train_pre = train_pre.replace(np.nan, 0)

    return train_pre


def create_submission(test_raw, model_vars, sub_file_name, model_file="rf_model.sav"):
    """ Function to preprocess & subset the test data, load a pre-trained model, & make preds on the new data.
    Output: Sorted df of predictions, in order or relevance / likelihood to book.
    Head of df (including 'scores') is displayed, while submission is written to 'submissions' folder."""
    # Process test data
    test_clean = preprocess_step(test_raw)
    # Load model
    model = pickle.load(open("models/" + model_file, 'rb'))
    # Get results
    test_set = test_clean[model_vars] if model_vars else test_clean
    test_pred_prob = model.predict_proba(test_set)
    # Take p value for boolian booking = 1
    booking_p = [b for a, b in test_pred_prob]

    # Create submission df
    submission = test_raw[['srch_id', 'prop_key']].copy()
    submission['score'] = booking_p
    # Sort according to competition rules
    sorted_sub = submission.groupby(['srch_id'])[['prop_key', 'score']].apply(
        lambda x: x.sort_values('score', ascending=False)).reset_index()
    final_sorted_sub = sorted_sub.drop(['level_1', 'score'], axis=1)

    # Export submission data
    final_sorted_sub.to_csv("submissions/" + sub_file_name, header=True, index=False)



    return sorted_sub.head()