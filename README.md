# icartt_read_and_merge
This set of functions provides all the functionality you need for reading in ICARTT files as pandas dataframes, and saving them as pickle files for quick access before data analysis. It can load in individual ICARTT files as a pandas dataframe, or can read in multiple ICARTT files from a directory and merging them together into a single pandas dataframe. It auto-recognizes times from ICARTT files, converts them to datetime objects and associates the appropriate time zone with them. There is functionality built in to merge to a single, standard time base, as well if you have files collected at different frequiences. It also parses metadata contained in the ICARTT file and builds a dictionary with this info for you, saves it along with the dataframe made. 

>Untested: This module has underlying funcationality to assess multileg flights if the icartts passed are passed as a list instead of a string. I'm not sure if it works or not.

**Requirements:** Python 3
* `numpy`
* `pandas`
* `os`
* `re`
* `sys`
* `csv`
* `matplotlib`
* `datetime`
* `mpu`


## Examples of Usage: 

### Example 1:  **Read in a single ICARTT file**  
Here we load in a single ICARTT file as a pandas dataframe and the meta data as a dict with info pulled from the ICARTT. 

```py
import datetime as dt 
import icartt_read_and_merge as ict

icartt_file='C:\\Users\\myusername\\Data\\WINTER\\WINTER-Merged1s_C130_20150303_R6_RF09.ict'

df, meta= ict.read_icartt(icartt_file) # Read in the icartt file as a pandas dataframe
```
We've got the raw data out now. But time is still in weird units. To deal with that, we can pass that dataframe to the built in `master_icartt_time_parser()` function which converts the classic ICARTT time columns to pandas datetime series, with timezone info included (if it recognizes it). It always spits out time columns in UTC time, so then we use the datetime module to convert the datetimes to Eastern Standard Time (where this data was collected). 
```py
# Convert the time columns to datetime objects (in UTC time). Give back df, and a list of time columns
df, times = ict.master_icartt_time_parser(df, icartt_file, quiet=True, remove_old_time=True)

# Convert times from UTC time to Eastern Standard Time. 
df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern')
```
Let's say we want to average all thsi data to get a 5 minute average (instead of having 1s data). We can use the built in `align2master_timeline()` function to do so as follows. We just set the start and end date, along without step and let the function know that the dataframe does not have a datetime index yet. I've set the option `quiet=False` so that it prints off a graph showing the original data vs the new averaged data so you can visually check what's happening here.  
```py
start_date=str(df.datetime.min()) # is '2015-03-03 00:06:55'
end_date=str(df.datetime.max()) # is '2015-03-03 08:20:22'
tm_step= 5*60 # timestep must be in seconds. 

# Reaverage this data set to a 5 minute average (instead of 1s). 
df= ict.align2master_timeline(df, start_date, end_date, tm_step, quiet=False,datetime_index=False)

df= df.set_index('datetime') # Now we can set the index the dataframe using time.
```
### EXAMPLE 2
**Read in a whole directory of ICARTT files from different flights and stack them all together on top of one another.**  To do this, we're using a single built in function, `icartt_merger`
```py
data_directory = "C:\\Users\\myusername\\Data\\WINTER\\WINTER_MERGE_60s\\"
output_directory = "C:\\Users\\myusername\\Data\\outputs\\"
output_filename = 'supermerge_winter_60s'
mode_input = 'Stack_On_Top'

df, meta = icartt_merger(data_directory, mode_input, output_directory=output_directory,output_filename=output_filename,     
                         prefix_instr_name=False)

```
This function does a lot of leg work on its own, looping over every file in the data directory, changing the times to datetimes, indexing the resulting dataframe by both flight # and datetime, and then putting them all into a single big dataframe that it saves in output directory as output filename. Lets take a peek at this function specifically and its possible inputs. 
```py
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
```

### Example 3 
**Take a bunch of ICARTT files from different instruments and merge them all together on the same time basis.** Similar to the last call, we're using the same function but its doing a merge on a totally different axis this time (horz instead of vert). This call would take all the ICARTT files in the data directory from different instruments and merge them "next" to one another after resampling them to a 30 minute time average convering the entire campaign timeline.  It will also append the instrument name as a prefix to the column names so if there are multiple measurements of "NO3" then you can tell which came from a PILS and which came from an AMS., etc. This call would save the reuslting dataframe it at the output directory as the outputfile name. 
```py
data_directory = "C:\\Users\\myusername\\Data\\SOAS\\Aerosol_contn\\"
output_directory = "C:\\Users\\myusername\\Data\\outputs\\"
output_filename = 'SOAS_contn_aero_stitch'
prefix_instr_name = True
mode_input = 'Merge_Beside'
master_timeline = ['2013-06-01 00:00:00', '2013-07-15 00:00:00', 1800]

df, meta = icartt_merger(data_directory, mode_input, master_timeline,
                         output_directory, output_filename,
                         prefix_instr_name)

```

