"""A script showing off some of the functionality of icartt_rea_and_merge."""

import icartt_read_and_merge as ict

# =============================================================================
#                             EXAMPLE #1
# =============================================================================
# Path to a single file
icartt_file = 'C:\\Users\\DATA\\icartt_name_20150223_R3_RF06.ict'

df_out, meta = ict.read_icartt(icartt_file)  # Read in the icartt file as df

# Get the time variables read as datetimes in UTC time, drop the old time col
df_fancy, times = ict.master_icartt_time_parser(df_out, icartt_file,
                                                quiet=True,
                                                remove_old_time=True)
# Get the beginning and end of this data
startdate = str(df_fancy.datetime.min())
enddate = str(df_fancy.datetime.max())
tm_step = 5 * 60  # Do a 5 minute average (tm_stepmust be in seconds)

# Re-sample and get a 5 minute average of the data.
df_5min = ict.align2master_timeline(df_fancy, startdate, enddate, tm_step,
                                    quiet=False)

# =============================================================================
#                          "MERGE_BESIDE" example;
# ============================================================================
data_directory = "C:\\Users\\myusername\\Data\\icartts\\"
output_directory = "C:\\Users\\myusername\\Data\\outputs\\"
output_filename = 'special_name_outfile'
prefix_instr_name = True
mode_input = 'Merge_Beside'
master_timeline = ['2013-06-01 00:00:00', '2013-07-15 00:00:00', 1800]

df, meta = ict.icartt_merger(data_directory, mode_input, master_timeline,
                             output_directory, output_filename,
                             prefix_instr_name)

# =============================================================================
#                          "STACK_ON_TOP" example;
# =============================================================================
data_directory = "C:\\Users\\myusername\\Data\\icartts\\"
output_directory = "C:\\Users\\myusername\\Data\\outputs\\"
output_filename = 'special_name_outfile'
mode_input = 'Stack_On_Top'

df, meta = ict.icartt_merger(data_directory, mode_input,
                             output_directory=output_directory,
                             output_filename=output_filename,
                             prefix_instr_name=False)
