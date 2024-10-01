"""
Plot painting retrieval results tables

Usage:
  plot_results.py <inputFile> 
  plot_results.py -h | --help
  -
  <inputFile>    File with the results of the scoring (csv file)
Options:
"""

from docopt import docopt
import pandas as pd
import dataframe_image as dfi
import numpy as np
import sys


if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
     
    input_file = args['<inputFile>']

    
    df = pd.read_csv(input_file, header=None)

    df.columns = ['Week', 'Measure', 'kVal', 'QS', 'Team', 'Method', 'f1','f2','f3','f4']


    styles = [dict(selector="caption",
                   props=[("text-align", "center"),
                          ("font-size", "120%"),
                          ("color", 'black'),
                          ("font-weight", 'bold')])]    
    
    # Get the different values of k
    k_vals = list(set(df['kVal'].tolist()))
    k_vals.remove(0)
    
    # Get the week
    weeks = list(set(df['Week'].tolist()))
    if len(weeks)!= 1:
        print ('Error: data from more one week!')
        sys.exit()
    week = weeks[0]
        
    # Get the different values of Query Set
    query_sets = set(df['QS'].tolist())


    for query_set in query_sets:
        # Query result    
        dfqs = df.loc[(df['QS'] == query_set) &
                     ((df['Measure'] == 'Q') | (df['Measure']=='QS') | (df['Measure']=='QN') | (df['Measure']=='QC'))].copy()
        # Remove unwanted columns
        dfqs.drop(columns=['Week','QS', 'f2', 'f3', 'f4'], inplace=True)
        
        if len(k_vals) > 1:
            print (k_vals)
            # https://stackoverflow.com/questions/17298313/python-pandas-convert-rows-as-column-headers
            df2 = dfqs.pivot_table(values=['f1'], index=['Measure','Team','Method'], columns='kVal')
            df2.reset_index(drop=False, inplace=True)
            # https://riptutorial.com/pandas/example/18695/how-to-change-multiindex-columns-to-standard-columns
            df2.columns = ['Measure','Team','Method']+[f'map@{kv}' for kv in k_vals]
        else:
            df2 = dfqs.copy()

        # Query result    
        dfq = df2.loc[df2['Measure'] == 'Q'].copy()
        dfq.drop(columns=['Measure'], inplace=True)

        # Sort by map, descending
        dfq.sort_values('map@1', inplace = True, ignore_index=True, ascending=False)

        
        # Format to 2 decimals
        df_styled = dfq.style.format(precision=2).hide(axis="index").set_caption(f'Query results for {query_set}').set_table_styles(styles)
        
        # Write the image with the table
        #https://stackoverflow.com/questions/35634238/how-to-save-a-pandas-dataframe-table-as-a-png
        dfi.export(df_styled,f'{week}_{query_set}_Q.png')

        # Images with no augmentations    
        dfna = df2.loc[df2['Measure'] == 'QN'].copy()
        if len(dfna) > 0:
            dfna.drop(columns=['Measure'], inplace=True)
            dfna.sort_values('map@1', inplace = True, ignore_index=True, ascending=False)
            df_styled = dfna.style.format(precision=2).hide(axis="index").set_caption(f'Query results (no augmentation): {query_set}').set_table_styles(styles)
            dfi.export(df_styled,f'{week}_{query_set}_QN.png')
                
        # Images with noise
        dfno = df2.loc[df2['Measure'] == 'QS'].copy()
        if len(dfno) > 0:
            dfno.drop(columns=['Measure'], inplace=True)
            dfno.sort_values('map@1', inplace = True, ignore_index=True, ascending=False)
            df_styled = dfno.style.format(precision=2).hide(axis="index").set_caption(f'Query results (noise): {query_set}').set_table_styles(styles)
            dfi.export(df_styled,f'{week}_{query_set}_QS.png')
                
        # Images with color changes
        dfco = df2.loc[df2['Measure'] == 'QC'].copy()
        if len(dfco) > 0:
            dfco.drop(columns=['Measure'], inplace=True)
            dfco.sort_values('map@1', inplace = True, ignore_index=True, ascending=False)
            df_styled = dfco.style.format(precision=2).hide(axis="index").set_caption(f'Query results (color changes): {query_set}').set_table_styles(styles)
            dfi.export(df_styled,f'{week}_{query_set}_QC.png')
    

        # Pixel Masks (PM)    
        dfpm = df.loc[(df['QS'] == query_set) & (df['Measure'] == 'PM')].copy()
        if len(dfpm) > 0:
            dfpm.drop(columns=['Week','Measure', 'kVal', 'QS', 'f4'], inplace=True)
            dfpm.sort_values('f3', inplace = True, ignore_index=True, ascending=False)
            df_styled = dfpm.style.format(precision=2).hide(axis="index").set_caption(f'Pixel masks: {query_set}').set_table_styles(styles)
            dfpm.rename(columns = {'f1':'Precision', 'f2':'Recall','f3':'F1'}, inplace = True)
            dfi.export(df_styled,f'{week}_{query_set}_PM.png')

        # Text Boxes (TB)    
        dftb = df.loc[(df['QS'] == query_set) & (df['Measure'] == 'TB')].copy()
        if len(dftb) > 0:
            dftb.drop(columns=['Week','Measure', 'kVal', 'QS'], inplace=True)
            dftb.sort_values('f4', inplace = True, ignore_index=True, ascending=False)
            df_styled = dftb.style.format(precision=2).hide(axis="index").set_caption(f'Text Boxes: {query_set}').set_table_styles(styles)
            dftb.rename(columns = {'f1':'Precision', 'f2':'Recall','f3':'F1', 'f4':'IoU'}, inplace = True)
            dfi.export(df_styled,f'{week}_{query_set}_TB.png')

        # Text Distance (TD)    
        dftd = df.loc[(df['QS'] == query_set) & (df['Measure'] == 'TD')].copy()
        if len(dftd) > 0:
            dftd.drop(columns=['Week','Measure', 'kVal', 'QS', 'f4'], inplace=True)
            dftd.sort_values('f1', inplace = True, ignore_index=True, ascending=True)
            dftd = dftd.astype({'f2':'int', 'f3':'int'})
            # Mean Text Distance
            dftd.rename(columns = {'f1':'mTD', 'f2':'Total lines','f3':'Checked lines'}, inplace = True)
            df_styled = dftd.style.format(precision=2).hide(axis="index").set_caption(f'Text distance: {query_set}').set_table_styles(styles)
            dfi.export(df_styled,f'{week}_{query_set}_TD.png')


        # Angular Error (F)
        dfae = df.loc[(df['QS'] == query_set) & (df['Measure'] == 'F')].copy()
        if len(dfae) > 0:
            dfae.drop(columns=['Week','Measure', 'kVal', 'QS', 'f3', 'f4'], inplace=True)
            dfae.sort_values('f1', inplace = True, ignore_index=True, ascending=True)
            # Mean Angular Error
            dfae.rename(columns = {'f1':'mAE', 'f2':'mIoU'}, inplace = True)
            df_styled = dfae.style.format(precision=2).hide(axis="index").set_caption(f'Angular Error: {query_set}').set_table_styles(styles)
            dfi.export(df_styled,f'{week}_{query_set}_AE.png')
