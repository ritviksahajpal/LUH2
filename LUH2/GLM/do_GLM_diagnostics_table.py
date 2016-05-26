import process_GLM
import constants
import pandas as pd

if __name__ == '__main__':
    # Create diagnostic table for each continent
    for idx, continent in constants.dict_conts.iteritems():
        # Ignore Antartica
        if continent == 'Antartica':
            continue

        process_GLM.do_diagnostics_table(region=idx, fname=continent + '_', map_region='continent')

    # Create diagnostic table for countries
    dict_cntr = pd.read_csv('../countries.csv').set_index('ID')['Name'].to_dict()
    process_GLM.do_diagnostics_table(region=840, fname=dict_cntr[840] + '_', map_region='country')

    # Global
    process_GLM.do_diagnostics_table(fname='Global_')
