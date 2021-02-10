"""Set of Functions that allow you to read in Icartt Files and Merge them."""
#  DESCRIPTION
#
#     This script provides all the functionality you need to read in
#     ICARTT data as a pandas dataframe & save it as a pickle for
#     quick usage in python scripts. Options are provided to merge
#     data from multiple icartt files into a single file, and to
#     remap the time averaging.
#
#  NOTES
#
#  * parser is based on icartt v2.0 spec. most convenient source, imo:
#    https://www-air.larc.nasa.gov/missions/etc/IcarttDataFormat.htm
#
#  * and spec reference on earthdata;s page:
#    https://cdn.earthdata.nasa.gov/conduit/upload/6158/ESDS-RFC-029v2.pdf
#
#  ACKNOWLEDGEMENTS
#    This modeule was modified from a really nice & much fancier module
#    (ornldaac_icartt_to_netcdf) written by mcnelisjj@ornl.gov
#    that was intended to convert ICARTT v2 files into netCDF.
#    Thanks to them for doing the bulk of the work
#
#  WRITTEN BY:
#    Dr. Jessica D.Haskins (jhaskins@alum.mit.edu)

import os
import re
import sys
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import mpu


def _warn(message: str = "( miscellaneous warning )"):
    """Print the input error message and exits the script with a failure."""
    # Args: message (str): warning message gets printed during eval and ignored
    print("   WARN: {}. Skipping.".format(message))


def _exit_with_error(message=str):
    """Print the input error message and exits the script with a failure."""
    sys.exit(print("ERROR: {} -- Abort.".format(message)))


def _crawl_directory(path: str, extension: str = None):
    """Crawl an input directory for a list of ICARTT files.\
     Parameters:------------------\
        path (str): full path to an input directory.\
        ext (str): An optional extension to limit search.\
    Returns:  A list of paths to data files (strings)."""
    selected_files = []  # Create an empty list

    for root, dirs, files in os.walk(path):  # Walk directory.
        for f in files:  # Loop over files,
            fext = os.path.splitext(f)[1]  # Get the extension.

            # If file matches input extension or if no extension given,
            if extension is None or extension == fext:

                # Join to root for the full path.
                fpath = os.path.join(root, f)
                # Add to list.
                selected_files.append(fpath)

    # Return the complete list.
    return selected_files


def _organize_standard_and_multileg_flights(DATA: dict):
    """Organize the Multileg flights & parse them."""
    # A regular expression catches the multi leg flight suffix.
    multileg_regex = re.compile('_L[0-9].ict')

    # A dictionary stores the output filename and legs as child list.
    flights = {}

    for ict in DATA['ICARTT_FILES']:  # Loop over all files in the directory.
        # If regular expression is not matched anywhere in string,
        if re.search(multileg_regex, ict) is None:
            # Add to list of "standard" flights (e.g. not a leg)
            flights[ict] = ict

        else:  # Else if regular expression is matched in string.
            # The output file won't have the suffix.
            output_filename = ict[:-7] + ".ict"

            # Add this file to the dict of multi-leg flights.
            if output_filename not in flights:
                flights[output_filename] = [ict]
            else:
                flights[output_filename].append(ict)

    return flights  # Return the organized flights as a dictionary.


def align2master_timeline(df: pd.DataFrame, startdt: str, enddt: str,
                          step_S: int, quiet: bool = True,
                          lim: int = None, datetime_index: bool = False,
                          tzf: str = 'UTC'):
    """Resample dataframes to appropriate timelines."""
    # Function to take a dataframe and appropriately remap it to a new time
    # index, considering the native sampling frequency as it is relative
    # to the desired new time. Writte 2/6/21, jessica d. haskins

    # --------------------------Inputs:-------------------------------------
    # df - dataframe, mustcontain column 'datetime'.
    # startdt, enddt = '2006-03-01 00:00:00', '2006-04-01 00:00:00'
    # step_S= 120 averaging step in seconds (120 for a 2 minute average).
    # quiet - Set to False to show sanity check on averaging.
    # lim -manually set the limit of # of points to include in an avg.
    # datetime index - Set to True if datetime is already an index of the df.
    if (datetime_index is False) and 'datetime' not in df:
        _exit_with_error(("Dataframe passed to align2master_timeline()",
                         "does not contain a column 'datetime'. "))

    # Make datetime an index and remove duplicates.
    print(datetime_index)
    if datetime_index is False:
        df['datetime'] = df['datetime'].dropna()
        df = df.set_index('datetime')  # Make the datetime an index.

    df = df[~df.index.duplicated()]  # remove any duplicates rows

    # Get the average native sampling frequency in total seconds:
    tseries = df.index.to_series()
    tseries = tseries.dropna()
    min_sep = int(np.round(tseries.diff().mean().total_seconds()))

    if quiet is False:
        print('Native Mean Time Sep. (s): ', str(min_sep) + 's')


    # If the native time seperation is less than X seconds, you'll
    # reindex it to our full date range, as close to native  freq as you can,
    # then take a roliing avg to get the X second avg on the time base we want
    if min_sep < step_S:
        dts = pd.date_range(startdt, enddt, freq=str(min_sep) + 's',
                            tz=tzf)

        if not lim:
            lim = np.round(4350 / min_sep)  # don't fill if collect>than 1H out
        dfn = df.reindex(dts, method='nearest', fill_value=np.NaN,
                         limit=int(lim))

        # Take a centered boxcar average around the X s avg.
        df_new = dfn.rolling(str(int(step_S)) + 's').mean().resample(str(step_S) + 's').mean()
    else:
        # The native sampling frequency is longer than X seconds, so just pull
        # the closest values along to fill our array.
        print(('WARNING: You have input an averaging frequency that is LESS'),
              ('than this instruments average native sampling frequency.'),
              ('This is dangerous and can lead to errors because it is NOT'),
              ('interpolating data, but rather "filling from nearest".'),
              ('Consider raising your input averaging time step.'))
        dts = pd.date_range(startdt, enddt, freq=str(step_S) + 's', tz='UTC')
        if not lim:
            # Don't fill if collected > than 1H 15 mins out from here.
            lim = np.round(4350 / min_sep)
            if lim <= 0:
                lim = 1  # If lim not set and too small, just use 1 point.
            
            df_new = df.reindex(dts, method='nearest', fill_value=np.NaN,
                                limit=int(lim))

    # Plot the Original Data & the Re- Mapped stuff so you can see if its good:
    if quiet is False:
        one = df.columns[0]
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.plot(df[one], label="original",
                color='orange', linewidth=1)
        ax.plot(df_new[one],
                label="re-mapped", color='blue', linewidth=1)
        # ax.set_xlim(df.index[200], df.index[400])
        # ax.set_ylim(34, 39)
        ax.legend()
        plt.show()

    return df_new


def _find_datelike_cols(df: pd.DataFrame, icartt_file: str,
                        quiet: bool = True):
    """Identify the date like columns in a dataframe from an icartt file."""
    # List of partial Strings to search columnNames for that will ID them as
    # time cols. ***NOTE:  If you are getting a persistent error about this
    # function, try adding to the list of tm_names. E.g. add other timezone
    # denotations. Setting quiet to False will print your col names and
    # what it is finding so you can decide what to add to this list.
    tm_names = ['utc', 'cst', 'cdt', 'local', 'lst', 'est', 'pst',
                'gmt'', lt', 'time_mid', 'central_standard',
                'eastern_standard', 'pacific_standard']

    print('All Column Names:', df.columns) if quiet is False else None

    # Identify all the names of time related columns in the dataframe.
    times = list()  # Empty list to contain columns with time-like names.
    for col in df.columns:
        if any(nm in col.lower() for nm in tm_names):
            times.append(col.lower())  # fill the list with those names.
            # Rename all time cols in lowercase to make string matches easier
            df.rename(columns={col: col.lower()}, inplace=True)

    # Make sure you haven't grabbed a day column for time by accident. Drop it.
    for h in range(0, len(times)):
        times.remove(times[h]) if 'day' in times[h] else None

    print('Original Time Columns Found:', times) if quiet is False else None
    # Return the dataframe with lowercase time names, a list of the time
    # columns found, and a list of time strings you searched for in the col
    # names (this is just for consistency when looping...)
    return df, times, tm_names


def _make_time_midpoint_cols(df: pd.DataFrame, tm_names: list, times: list,
                             quiet: bool = True):
    """Take start/stop times from datelike cols and turn them into midpts."""
    # Get start/stop pairs for time to make a midpoint if you can find both.

    # Create a dict for time names & their ID'd "type" for scanning later.
    nn_times = {}
    for j in range(0, len(tm_names)):  # Check each time zone name
        start_j = None  # reset on each loop through a dif time_nm
        stop_j = None
        has_tm = None
        for i in range(0, len(times)):  # Check each time col name.
            # Assign start/stop pair variables.
            if (tm_names[j] in times[i]) and ('start' in times[i]):
                start_j = times[i]

            if (tm_names[j] in times[i]) and \
               ('stop' in times[i] or 'end' in times[i]):
                stop_j = times[i]

            # Or just let us know if this "time" col exists.
            if (tm_names[j] in times[i]):
                has_tm = times[i]

        # You found a start/stop pair for this time name.
        if (start_j is not None) and (stop_j is not None):
            # Make a new column in the data frame that is the midpoint.
            df[tm_names[j] + '_mid'] = (df[start_j] + df[stop_j]) / 2

            # Remove the start & stop pair of time from the larger df
            df = df.drop(columns=[start_j, stop_j])

            # Update list of column names with times to reflect above.
            times.append(tm_names[j] + '_mid')  # add midpoint name
            times.remove(start_j)  # remove oldies
            times.remove(stop_j)

        elif has_tm is not None:
            # Populate dict associating its actual timename and its "type" so
            # we can choose between times to use as master later if we want.
            nn_times[has_tm] = 'time_' + tm_names[j]

            # Didn't find a start/stop pair but do have a col with this tm nm.
            # df.rename(columns={has_tm: 'time_' + tm_names[j]},
            #          inplace=True)  # rename it so we know what it is called

            # Update list of column names with times to reflect above.
            # times.append('time_' + tm_names[j])  # add new name
            # times.remove(has_tm)  # remove old name

    print('Time cols After Mid_Point Assign:',
          times) if quiet is False else None

    return df, times, tm_names, nn_times


def _pick_a_single_time_col(times: list, nn_times: dict, quiet: bool = True):
    """Decide which time column you prefer to use a indx if you got lots."""
    # Create dictionary for "picking" which time variable we prefer to use.
    # Change the rank for preferences here. Lower = more desirable.
    pref_time = {'time_utc': 1, 'utc_mid': 2,
                 'time_est': 3, 'est_mid': 4,
                 'time_cst': 5, 'cst_mid': 6,
                 'time_cdt': 7, 'cdt_mid': 8,
                 'time_pst': 9, 'pst_mid': 10,
                 'time_local': 11, 'local_mid': 12,
                 'time_lt': 13, 'lt_mid': 14,
                 'time_lst': 15, 'lst_mid': 16,
                 'time_time_mid': 17, 'time_mid_mid': 18}

    # Create dictionary to associate a "time" column name with its timezone.
    tz_info = {'time_utc': 'UTC', 'utc_mid': 'UTC',
               'time_est': 'EST5EDT', 'est_mid': 'EST5EDT',
               'time_cst': 'US/Central', 'cst_mid': 'US/Central',
               'time_cdt': 'CST6CDT', 'cdt_mid': 'CST6CDT',
               'time_pst': 'PST8PDT', 'pst_mid': 'PST8PDT'}

    pref_arr = list()  # empty list to contain pref rank of "time cols"
    tz_arr = list()  # Empty list to contain the timezone of the "time cols"

    for n in range(0, len(times)):  # Get an array of # preferences for "times"
        # Get the nickname of the time col if its not a mid point.
        check_if_NOT_mid = nn_times.get(times[n], None)
        if check_if_NOT_mid is None:  # has no nickname,it is a midpoint.
            pref_arr.append(pref_time.get(times[n], 100))
            tz_arr.append(tz_info.get(times[n], 100))
        else:  # has a nickname, pass that to get preference.
            pref_arr.append(pref_time.get(check_if_NOT_mid, 100))
            tz_arr.append(tz_info.get(check_if_NOT_mid, 100))

    if min(pref_arr) == 100:
        _exit_with_error(('Time columnName in the "pick_a_single_time_col()"'),
                         ('function could not be properly identified. Try'),
                         ('running the call to function with quiet=False'),
                         ('to see debug output.'))

    # Pick one of the time columns based on your ranked preferences.
    time_pref = times[(pref_arr == min(pref_arr))]  # Name of the pref time col
    tz_pref = tz_arr[(pref_arr == min(pref_arr))]  # Timeezone of pre time col

    if tz_pref == 100:
        _exit_with_error(('Unable to ID timezone in pick_a_single_time_col()'),
                         ('function.Try running the call to function with'),
                         ('quiet=False to see debug output.'))

    print('Pref time col:', time_pref, ' in', tz_pref,
          'Timezone') if quiet is False else None

    bad_times = times  # duplicate list of all times
    bad_times.remove(time_pref)  # and drop all non pref times.

    # Return name of preffered column its timezone and the names of all the
    # columns that you don't want to use.
    return time_pref, tz_pref, bad_times


def icartt_time_to_datetime(df: pd.DataFrame, yr, mon, day, time_col: str,
                            tz_pref: str, remove_old_time: bool = True):
    """Convert seconds since midnight (icartt time col) 2 datetime obj col."""
    # Takes yr mon day in string or int form. Get to ints if strings passed.
    yr = int(yr) if type(yr) == str else yr
    mon = int(mon) if type(mon) == str else mon
    day = int(day) if type(day) == str else day

    # Add "timedelta" of seconds since midnight to the date the icarrt
    # file started on (typically in the file name).
    datetime_col_i = datetime.datetime(yr, mon, day) + \
        pd.to_timedelta(df[time_col], 's')
        
    # Teach this tz unaware pandas series its native timezone.
    datetime_col = datetime_col_i.dt.tz_localize(tz_pref)  # now it has a tz

    # Convert from native tz to UTC time.
    datetimecol_in_UTC = datetime_col.dt.tz_convert('UTC')

    df['datetime'] = datetimecol_in_UTC  # Create new column with date in UTC.

    if remove_old_time is True:  # Drop the old col from the df if asked
        df = df.drop(columns=time_col)

    return df


def master_icartt_time_parser(df: pd.DataFrame, icartt_file: str,
                               quiet: bool = True, remove_old_time:
                                   bool = True):
    """Identify the date like columns, convert to TZ aware datetimes)."""
    # Identify the time-like columns in this dataframe:
    df, times, tm_names = _find_datelike_cols(df, icartt_file, quiet)

    # Take Start/Stop pairs of time cols and convert them to midpoint times.
    df, times, tm_names, nn_times = _make_time_midpoint_cols(df, tm_names,
                                                             times, quiet)

    # Pick which time var to use and which time zone its in.
    time_pref, tz_pref, bad_times = _pick_a_single_time_col(times,
                                                            nn_times, quiet)
    if remove_old_time is True:  # Remove the non-preferred times
        df = df.drop(columns=bad_times)
        # you've dropped all other names from the df.
        times = list([time_pref])
    else:
        # you haven't dropped anything.
        times = list([time_pref, bad_times])

    # Get the date this data was collected on from the icarttfile name passed.
    date_full = re.search(r'\d{4}\d{2}\d{2}', icartt_file).group(0)
    yr = date_full[0:4]
    mm = date_full[4:6]
    dd = date_full[6:8]

    # Tell the people which variable was chosen as "datetime".
    print('The time variable chosen to be converted to "datetime" is:',
          time_pref)
    
    # Convert the preffered time column to a column named 'datetime' and drop
    # all the other time columns from the larger dataframe.
    df = icartt_time_to_datetime(
        df, yr, mm, dd, time_pref, tz_pref, remove_old_time)

    # And update the times list to remove the old time and add "datetime"
    times.append('datetime')
    if remove_old_time is True:
        times = times.remove(time_pref)

    return df, times


def char_cleaner(mystring, ignore: list = []):
    """Clean up gross strings from weird characters."""
    after = mystring.strip()  # strip all leading/trailing whitespace

    # Then, replace common representations with a word.
    after = after.replace('%', 'percent')
    after = after.replace('_+_', '+')
    after = after.replace('-->', 'to')

    # A list of bad chars we don't want in our string.
    bad_chars = [' ', ',', '.', '"', '*', '!', '@', '#', '$', '^', '&',
                 '(', ')', '=', '?', '/', '\\', ':', ';', '~', '`', '<',
                 '>', ']', '[', '{', '}']

    for i in range(0, len(bad_chars)):
        if bad_chars[i] not in ignore:  # don't replace chars they want
            after = after.replace(bad_chars[i], '_')

    return after


def _build_meta_dict(icartt_file: str, meta: dict = {}, flt_num: int = None):
    """Take and combines metadata from different\
    icartt files into a dictionary. So you can access metadata from an\
    individual icartt file by typing in its instrument name & flt #."""
    # If meta empty, initialize the dict.
    if bool(meta) is False:
        meta = {'Instruments': {}, 'Data_Info': {}, 'Instrument_Info': {},
                'PI_Info': {}, 'Uncertainty': {}, 'Revision': {},
                'Stipulations': {}, 'Institution_Info': {}}
    # Open the file.
    with open(icartt_file, "r") as f:  # Get number of header rows
        header_row = int(f.readlines()[0].split(",")[0]) - 1

    with open(icartt_file, "r") as f:
        reader = csv.reader(f)
        ln_num = 0  # intitalize line counting var.
        for row in reader:
            line = " ".join(row)  # read line by line.

            # Icartt splits headers with a ":", use that to split them.
            before, sep, after = line.rpartition(":")
            after = char_cleaner(after, ignore=':')  # Pass to string cleaner.

            # First 3 lines have set parameters in ICARTT Files.
            if ln_num == 1:
                PI = after
            if ln_num == 2:
                Institution = after
            if ln_num == 3:
                Instrument = after
            
            # Once you know the instrument, you can start to build the dict
            # (becase we are using the instrument part of the dict index we
            # can't do it until we get to this line number. )
            if ln_num == 3:
                meta['PI_Info'][Instrument] = PI
                meta['Institution_Info'][Instrument] = Institution
                if flt_num is not None:
                    meta['Instruments'][flt_num] = Instrument
                else:
                    meta['Instruments'] = Instrument

            # The rest of the meta data is on arbitrary line #s based on how
            # much info the author of the ICARTT included, so just parse the
            # string to ID which row that is contained on. Then,  append info
            # from  this file into the meta dictionary indexed on the
            # instrument name
            if 'DATA_INFO' in before:
                meta['Data_Info'][Instrument] = after
            if 'UNCERTAINTY' in before:
                meta['Uncertainty'][Instrument] = after
            if 'REVISION' in before:
                meta['Revision'][Instrument] = after
            if 'INSTRUMENT_INFO' in before:
                meta['Instrument_Info'][Instrument] = after
            if 'STIPULATIONS_ON_USE' in before:
                meta['Stipulations'][Instrument] = after

            if ln_num > header_row - 1:
                break  # top once you reach data.
            ln_num = ln_num + 1

    return meta  # return dictionary with this info.


def read_icartt(icartt_file: str, flt_num: int = None, meta: dict = {},
                instr_name_prefix: bool = False, add_file_no: bool = False):
    """Parse a single ICARTT file to a pandas dataframe."""
    # Get the header row number from the ICARTT.
    with open(icartt_file, "r") as f:
        header_row = int(f.readlines()[0].split(",")[0]) - 1

    # Parse the table starting where data begins (e.g. after the header).
    df = pd.read_csv(icartt_file, header=header_row, delimiter=",")
    
    # Set possible error values to NaNs.
    df.replace(-9, np.NaN, inplace=True)
    df.replace(-99, np.NaN, inplace=True)
    df.replace(-999, np.NaN, inplace=True)
    df.replace(-9999, np.NaN, inplace=True)
    df.replace(-99999, np.NaN, inplace=True)
    
    # Strip leading/tailing white space around variable names
    df.columns = [c.strip() for c in list(df.columns)]

    # Build/ append metadata from ICARTT to a dictionary file.
    meta = _build_meta_dict(icartt_file, meta, flt_num)

    # If instr_name_prefix is set to True, add the instrument name as a
    # prefix to the column names in the super merge dataframe so you know
    # which instrument collected that data (useful if multiple instruments
    # measure "NO3" and named them all "NO3". If set to false, then
    # you'd have duplicate column names in the resulting super-merge dataframe)
    if instr_name_prefix is True:
        if flt_num is not None:  # Indexed by flt # if more than 1 icartt file.
            df = df.add_prefix(meta['Instruments'][flt_num] + '_')
        else:  # Not indexed by flt # if only  1 icartt file.
            df = df.add_prefix(meta['Instruments'] + '_')

    if add_file_no is True:
        # Create a column same length as data that contains the file #
        sz = len(df[df.columns[0]])  # get appropriate length
        fnum_arr = np.full(shape=sz, fill_value=flt_num, dtype=np.int)
        df['Flight_N'] = fnum_arr

    return df, meta  # dataframe with data, and df with metadata


def _read_icartt_multileg(icartt_file: str, flt_num: int = None, meta:
                          dict = [], instr_name_prefix: bool = True):
    """Parse multi-leg icartt files, combine into single df."""
    # Sort the list of input ICARTTs.
    icartts = sorted(icartt_file)

    df = None  # Set an empty merged flight data data frame for this leg

    for ict in icartts:  # Loop over the dif ICARTT Legs
        # Parse individual file
        df_i, meta_i = read_icartt(ict, flt_num=int(flt_num), meta=meta,
                                   instr_name_prefix=instr_name_prefix)
        meta = meta_i  # update metadata file... gets appended upstream.

        # If df for merged is still None, set to first iter. Otherwise append.
        if df is None:
            df = df_i
        else:
            df = df.append(df_i, ignore_index=True)

    return df  # Return the merged df


def _main_loop_parse_flights(DATA: dict):
    """Looper for parsing indv flights in a directory."""
    # Make groupings of standard and multileg flights.
    DATA['FLIGHTS'] = _organize_standard_and_multileg_flights(DATA)

    ct = 1  # Initialize number of flights you're looping over.

    # Loop over the ICARTT files in the data directory.
    for flight, icartt in DATA['FLIGHTS'].items():

        # Tell the people which file you're processing:
        print("\n - [ {} / {} ] {} ".format(
            ct, len(DATA['FLIGHTS']), os.path.basename(flight)))

        if int(ct) == 1:  # Initialize empty dictionary to store metadata
            meta = {'Instruments': {}, 'Data_Info': {}, 'Instrument_Info': {},
                    'PI_Info': {}, 'Uncertainty': {}, 'Revision': {},
                    'Stipulations': {}, 'Institution_Info': {}}

        # If the type(icartt) is a list, must be MULTI-LEG flight.
        if type(icartt) is list:
            # Handle multileg flight so it is merged to single df.
            df_data, new_meta = _read_icartt_multileg(icartt, flt_num=int(ct),
                                                      meta=meta,
                                                      instr_name_prefix=DATA
                                                      ['PREFIX_OPT'])
            meta = new_meta  # Update the metadata dict to be the new one.

        # Else if the type(icartt) is a string, parse SINGLE flight (not leg)
        elif type(icartt) is str:
            add_file_no = False
            add_file_no = True if DATA['MODE'] == 'Stack_On_Top' else None

            # Call the parse_icartt_table directly, pass existing meta dict.
            df_data, new_meta = read_icartt(icartt, flt_num=int(ct),
                                            meta=meta, instr_name_prefix=DATA
                                            ['PREFIX_OPT'],
                                            add_file_no=add_file_no)
            meta = new_meta  # Update the metadata dict to be the new one.

        # Parse the string date columns in the indv icartt, and pick which
        # one to use, then convert it to a datetime object.
        df_data, times = master_icartt_time_parser(df_data, icartt,
                                                    quiet=True,
                                                    remove_old_time=True)

        if DATA['MODE'] == 'Merge_Beside':
            # To merge beside, must get all dfs must be on the SAME
            # time axis... so align them to a master timeline.
            # Aligns each file after its opened.
            tmln = DATA['MSTR_TMLN']  # wouldn't want to index twice!
            df_data = align2master_timeline(df_data, tmln[0], tmln[1],
                                            tmln[2], quiet=True,
                                            datetime_index=False)
            # Should come out with a datetime index we can merge along!

        if ct == 1:  # If first icartt file, make the larger dataframe!
            df_all = df_data
        else:
            if DATA['MODE'] == 'Stack_On_Top':
                # For all subsequent loops append the new one UNDER the old df
                df_all = pd.concat([df_all, df_data], ignore_index=True)
            else:
                # For all subsequent loops append new columns to existing df
                # along the same index.
                df_all = pd.concat([df_all, df_data], axis=1)

        ct += 1  # Update the counting variable.

    # Check if the User wants us to align the Stacked data to a master timeline
    # Aligns AFTER all icartts have been loaded in.
    if (DATA['MODE'] == 'Stack_On_Top') and (bool(DATA.get('MSTR_TMLN'))
                                             is True):
        tmln = DATA['MSTR_TMLN']  # wouldn't want to index twice!
        df_all = align2master_timeline(df_all, tmln[0], tmln[1], tmln[2],
                                       quiet=True, datetime_index=False)
    elif (DATA['MODE'] == 'Stack_On_Top'):  # Make file # and datetime indexes.
        df_all = df_all.set_index(['datetime', 'Flight_N'])

    return df_all, meta


def _handle_input_configuration(DATA: dict):
    """Make sure user passed appropriate input."""
    print('1. Input ICARTT directory:' + DATA['DIR_ICARTT'])

    # 1. Ensure ICARTT directory is valid.
    if not os.path.isdir(DATA['DIR_ICARTT']):
        _exit_with_error("Input ICARTT directory is invalid.")

    # 2. Ensure that directory  has ICARTT files in it. return  list.
    DATA['ICARTT_FILES'] = _crawl_directory(DATA['DIR_ICARTT'])

    if len(DATA['ICARTT_FILES']) == 0:
        # If no icartts were found, exit and notify user.
        _exit_with_error("No ICARTT files found in the input directory.")
    else:  # Else, inform on the number of ICARTT files
        print(" - Found [ {} ] ICARTTs.\n".format(len(DATA['ICARTT_FILES'])))

    # 3. Check that a valid mode has been passed
    valid_modes = ['Stack_On_Top', 'Merge_Beside']
    if DATA['MODE'] not in valid_modes:
        _exit_with_error(("Input mode entered is invalid."),
                         ("Valid Options are:" + valid_modes))

    # 4. Check that master_timeline info has been provided if necessary.
    if DATA['MODE'] == 'Merge_Beside':
        if bool(DATA.get('MSTR_TMLN')) is False:
            _exit_with_error(("For input mode 'Merge_Beside', input for "),
                             ("MSTR_TMLN is also needed."))

    return DATA  # give input back with the list of icartts now included.


def icartt_merger(data_directory: str,
                  mode_input: str,
                  master_timeline: list = [],
                  output_directory: str = '',
                  output_filename: str = 'icartt_merge_output',
                  prefix_instr_name: bool = True):
    """Merge a directory of icarrts into a pandas dataframe & save as a pkl.
    
    # ========================================================================
    # ========================    INPUTS   ===================================
    # ========================================================================
    #
    #   (1) data_directory - A string containing the absolute path to a folder
    #                      which contains all the individual icartt files that
    #                      you wish to merge together.
    #
    #   (2) mode_input - A string describing HOW you would like to merge these
    #                    icartt files in the data_directory together. Only 2
    #                    valid options are supported right now.  Either
    #                    "Stack_On_Top" or "Merge_Beside".
    #
    #     "Stack_On_Top": Each icartt file is for a different date,
    #      but contains data from multiple instruments or mutltiple
    #      measurements and  you want that data in a single
    #      file (e.g. indexed by time, and File/Flight #). Contents of
    #      individual icarrt files will be "stacked on top" of one another.
    #
    #     "Merge_Beside": Each icartt file is for the entire
    #      sampling period, but contains different measurements.
    #      You want to have all of these differnt measurments
    #      on the same time base, throughout the whole period. The contents
    #      of each icartt file will be "merged beside" one another.
    #
    #   (3) master_timeline - OPTIONAL if "Stack_On_Top", required if
    #                        "Merge_Beside". It is list with 3 items:
    #
    #       -  Startdate_str:  A string containing the start date of the
    #                        "mastertimeline" that all data  will be
    #                        merged to.Format is 'YYYY-MM-DD HH:MM:SS'
    #       -  Enddate_str:  A string containing the end date of the
    #                        "mastertimeline" that all data  will be
    #                        merged to. Format is 'YYYY-MM-DD HH:MM:SS'
    #       - Averaging_Step: An integer that is the number of seconds
    #                        for each timestep in between startdate and
    #                        end date. So 120 for a 2 minute average.
    #
    #    (4) output_drectory - OPTIONAL string containing the  abs path where
    #                         the output file will be written. If not set, the
    #                        output will be to stored in the  input data_dir.
    #
    #    (5) ouptut_filename - OPTIONAL string containing what you'd like the
    #                         output file to be called (not including its
    #                         extension). Default is 'icartt_merge_output'
    #
    #    (6) prefix_instr_name - OPTIONAL boolean value indicating whether you
    #                         would like to append the instrument name
    #                         contained in the icartt file to all the var
    #                         names. Default is True since when merging
    #                         icartt files it is common to have some PI's
    #                         measuring the same items & naming them the same.
    #
    # ========================================================================
    """
    # Format the input for easier referencing.
    inputs = {'DIR_ICARTT': data_directory,
              'DIR_OUTPUT': output_directory,
              'O_FILENAME': output_filename,
              'MODE': mode_input,
              'PREFIX_OPT': prefix_instr_name,
              'MSTR_TMLN': master_timeline}

    # Make sure you got appropriate inputs from the user, retrieve icartt files
    DATA = _handle_input_configuration(inputs)
    
    # Loop through parsing the flights & collecting them in a single dataframe.
    df, meta = _main_loop_parse_flights(DATA)
    
    # Save the Output.
    filename = DATA['DIR_OUTPUT'] + DATA['O_FILENAME'] + '.pkl'
    df.to_pickle(filename)
    
    # Save the metadata to a picke as well.
    filename_meta = DATA['DIR_OUTPUT'] + DATA['O_FILENAME'] + '_meta.pickle'
    mpu.io.write(filename_meta, meta)
    
    # Tell the people where you saved it.
    print('Output dataframe and metadata saved at:' + filename)
    
    return df, meta
