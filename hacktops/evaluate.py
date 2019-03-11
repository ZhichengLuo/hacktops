import pandas as pa

def recall_tops(df_tops_true, df_tops_pred, tolerance = 4):    
    
    if set(df_tops_true.columns) == set(df_tops_pred.columns) :
        concat_df = df_tops_true.copy()
        for col in df_tops_pred.columns:
            concat_df[col+"_pred"] = df_tops_pred[col]             
        tp = 0
        p = 0
        mae = 0
        for col in df_tops_true:   
            diffname = "{0}_ae".format(col)
            tpname = "{0}_tp".format(col)
            p += concat_df[col].count()          
            concat_df[diffname] = concat_df[col]-concat_df[str(col + "_pred")]        
            concat_df[diffname] = concat_df[diffname].abs()
            concat_df[tpname] = concat_df[diffname] <= tolerance 
            tp += concat_df[tpname].sum()
            mae += concat_df[diffname].sum()     
        return tp/p, mae/p, concat_df
    else :
        print("the tops columns are not valid")        
    return None,None,None
