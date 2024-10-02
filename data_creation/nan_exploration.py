import pandas as pd


def nan_thresholds(df):
    nan_ranges = {}

    for col in df.columns:
        nan_ranges[col] = []
        in_nan_range = False
        start_idx = None

        for i in range(len(df)):
            if pd.isna(df[col].iloc[i]) and not in_nan_range:
                start_idx = i
                in_nan_range = True
            elif not pd.isna(df[col].iloc[i]) and in_nan_range:
                end_idx = i - 1
                nan_ranges[col].append((start_idx, end_idx))
                in_nan_range = False
            elif pd.isna(df[col].iloc[i]) and i == len(df) - 1:
                end_idx = i
                nan_ranges[col].append((start_idx, end_idx))

        # If the column ends with NaNs
        if in_nan_range:
            nan_ranges[col].append((start_idx, len(df) - 1))

    return nan_ranges


def find_common_nan_ranges(nan_ranges):
    common_ranges = []

    # Convert the nan ranges to sets of indices
    sets_of_nan_indices = []
    for col, ranges in nan_ranges.items():
        nan_indices = set()
        for start, end in ranges:
            nan_indices.update(range(start, end + 1))
        sets_of_nan_indices.append(nan_indices)

    # Find intersection of all sets of indices
    common_nan_indices = set.intersection(*sets_of_nan_indices)

    # Convert the set of common indices back to ranges
    if common_nan_indices:
        sorted_indices = sorted(common_nan_indices)
        start_idx = sorted_indices[0]
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] != sorted_indices[i - 1] + 1:
                common_ranges.append((start_idx, sorted_indices[i - 1]))
                start_idx = sorted_indices[i]
        common_ranges.append((start_idx, sorted_indices[-1]))

    return common_ranges


def find_common_nan_indices(df, col_name):
    # Ensure the column name exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame")

    # Find the indices where the specified column has NaN values
    col_nan_indices = df.index[pd.isna(df[col_name])].tolist()

    # Find the indices where all other columns also have NaN values
    common_nan_indices = []
    for idx in col_nan_indices:
        if pd.isna(df.loc[idx]).all():
            common_nan_indices.append(idx)

    return common_nan_indices


def find_nan_with_non_nan_in_others(df, col_name):
    # Ensure the column name exists in the DataFrame
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in DataFrame")

    # Find the indices where the specified column has NaN values
    col_nan_indices = df.index[pd.isna(df[col_name])].tolist()

    # Find the indices where at least one other column does not have NaN values
    mixed_nan_indices = []
    for idx in col_nan_indices:
        # Check if any other column has a non-NaN value at this index
        if df.drop(columns=[col_name]).loc[idx].notna().any():
            mixed_nan_indices.append(idx)

    return mixed_nan_indices


# Find chunks of NaN and missing values
df = pd.read_csv("./aggr_10_min.csv")
nan_ranges = nan_thresholds(df=df)

'''for r in nan_ranges:
    print(f'{r}: {nan_ranges[r]}')'''

# Find common NaN ranges
common_nan_ranges = find_common_nan_ranges(nan_ranges=nan_ranges)
print("Common NaN Ranges:", common_nan_ranges)

sequence_length = 10

total_nan_entries = 0
useless_timeseries = 0
for start, end in common_nan_ranges:
    diff = end-start
    total_nan_entries += diff + 1
    if diff > sequence_length:
        useless_timeseries += 1

print(f'Nan dataset         : {total_nan_entries}')
print(f'Non-nan dataset     : {df.shape[0]-total_nan_entries}')
print(f'Useless timeseries  : {useless_timeseries}')

common_nan_indices = find_common_nan_indices(df, 'fuelVolumeFlowRate')
print(f'#Common NaN Indices : {len(common_nan_indices)}')

mixed_nan_indices = find_nan_with_non_nan_in_others(df, 'fuelVolumeFlowRate')
print(f'#Mixed NaN Indices  : {len(mixed_nan_indices)}')