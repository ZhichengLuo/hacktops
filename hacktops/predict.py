import pandas as pa

def predict( df_logs, df_postions = None, tops = ["NIOBRARA","CODELL","SUSSEX"]):
    """ Return the tops found at specific depth.

    Parameters
    ----------
    df_logs : Pandas DataFrame

    df_position : Optional Pandas DataFrame
    location of the wells 

    returns 
    -------
    df_tops_pred : Pandas DataFrame
    
    The format must be 

    	    A	B
    well_0	NaN	NaN
    well_1	NaN	NaN
    well_2	NaN	NaN

    Where the index is the list of wells fromw df_logs and colums are the depths of the tops to detect
    """
            
    # get the list of the wells from the input logs
    wells = list(df_logs["wellName"].unique())

    # Create the tops dataframe to output 
    df_tops_pred = pa.DataFrame(index = wells,columns = tops)

    #PUT ML CODE HERE TO PREDICT THE TOPS

    return df_tops_pred
