# -*- coding: utf-8 -*-

"""
SPDX-FileCopyrightText: Phillip Wendt

SPDX-License-Identifier: MIT
"""
import os
from oemof.solph import views
import pandas as pd
from mtress.run_mtress import run_mtress
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import yaml
import time

## Variables to declare

# --- Photovoltaic Panels ---
# Square meters of PV panes
pv_min = 10
pv_max = 50
pv_step = 10
# cost per qm PV:
kwp_per_qm = 0.2
cost_per_qm = 400 # 2000€/kWp
# efficiency:
pv_eff = 0.25
# -----

# ---Electric Storage---
# Capacity in kWh
storage_min = 1
storage_max = 40
storage_step = 10
# -----


# how many days to run model?
start_day = 1
days_to_run = 365
#infer_last_interval = True
#pause before first run
pause_time = 5

# enable plots?
enable_mg_plots = False

# name of output file for analysis
# (Change if you simulate a different month! Files with identical names get overwritten!)
results_filename = 'results_mini_grid_year_0.10.csv'
# This below is the results file for the dynamic plots, uncomment it to see the flows of your chosen configuration in detail
#results_filename_dyn = 'DYN_results_mini_grid_Jan.csv'
############################################

# days times hours per day,times intervals per hour
first_time_step = int(round((start_day*24)-23)) #  multiply times 24 h/day IF you want to run starting from other than day 1
last_time_step = first_time_step + int(round((days_to_run * 24)))

# Hard-coded locations of input files (DO NOT CHANGE)
dir_path = os.path.dirname(os.path.realpath(__file__))
yaml_file_name = os.path.join(dir_path, "minigrid.yaml")
radiation_file = os.path.join(dir_path, "radiation_Wm-2.xls")
input_file = os.path.join(dir_path, "input_demand.xls")


# Functions to extract data from files and to run model

def calculate_radiation_data():
    # Interpolates the radiation data supplied in a file to 15 min intervals, adapting the data to the format used in
    # MTRESS in the process.
    radiation_data = pd.read_excel(os.path.join(dir_path, radiation_file), parse_dates=True)
    #radiation_data = pd.read_csv(radiation_file, comment='#', sep=',', parse_dates=True, skiprows=8, nrows=8760)
    radiation_data['time'] = pd.to_datetime(radiation_data['time'], format='%Y%m%d:%H11')
    # interpolate radiation data to 15min
    radiation_data = radiation_data.set_index(radiation_data['time'])#.resample('15min').interpolate(method='linear')
    radiation_data.to_csv(os.path.join(dir_path, "radiation_data.csv"))
    #timeindex= radiation_data.set_index(pd.DatetimeIndex(radiation_data['time']))
    return radiation_data  #, timeindex


def calulate_pv_generation(radiation_data, j):
    # calculate pv generation based on the radiation data calculated before, with respect to kwp, efficiency of pv,
    # square meters.
    pv_generation = pd.DataFrame()
    pv_generation['kW'] = (((
            radiation_data['G'] * pv_eff * j/1000)))  # in kWh

    # save pv generation to csv
    pv_generation.to_csv(os.path.join(dir_path, "pv_generation.csv"))
    
    with open(yaml_file_name) as p:
        doc = yaml.load(p, Loader=yaml.FullLoader)
    pvcap = j*kwp_per_qm # as kWp
    doc['pv']['nominal_power'] = pvcap
    with open(yaml_file_name, 'w') as p:
        yaml.safe_dump(doc, p, default_flow_style=False) 

def read_input_data():
    # reads the input data, changing the format to datetime and converting the demand time series to MW
    # in the process.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    csv_data = pd.read_excel(os.path.join(dir_path, input_file), parse_dates=True)

    csv_data['time'] = pd.to_datetime(csv_data['time'],format='%Y%m%d:%H11')
    #csv_data = csv_data.set_index('time').resample('15min').interpolate(method='linear').drop(columns=['Time'])
    #csv_data['el_1'] = csv_data['heating']   # in kW
    #csv_data['dhw'] = csv_data['dhw'] # in kW
    csv_data.to_csv(os.path.join(dir_path, "input_demand.csv"))

    # Now reads input csv file. Aggregates demand of both heating and dhw to calculate power demand. Save results as
    # dataframe

    #csv_data = pd.read_csv(os.path.join(dir_path, "input_demand.csv"),
   #                        comment='#', sep=',', parse_dates=True)

    aggregated_kw = pd.DataFrame()
    aggregated_kw['Time'] = pd.to_datetime(csv_data['time'], format='mixed', utc=True, )

    aggregated_kw.set_index('Time')
    aggregated_kw['kW_el_demand'] = (csv_data['el_1'] + csv_data['el_2'])
    aggregated_kw['kW_th_demand'] = 0
    aggregated_kw['kW_dhw_demand'] = 0
    aggregated_kw.set_index(('Time'), inplace=True) #- commenting this line changed the error!!!
    aggregated_kw.to_csv(os.path.join(dir_path, "aggregated_demand.csv"))


def calculate_storage_cap(i):
    # calculate storage size and save it to the YAML

    with open(yaml_file_name) as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
    cap = i # in kWh
    doc['battery']['capacity'] = cap
    doc['battery']['power'] = cap/5
    with open(yaml_file_name, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)


def calculate_pareto(df):
    df = df.sort_values(by=['co2', 'capex'])

    # Initialize an empty list to store the Pareto front solutions.
    pareto_front = []

    # Iterate through rows and add non-dominated points to the Pareto front.
    for index, row in df.iterrows():
        non_dominated = True
        to_remove = []
        for i, existing_row in enumerate(pareto_front):
            if (
                    (row['co2'] >= existing_row['co2'] and row['capex'] >= existing_row['capex']) or
                    (row['co2'] > existing_row['co2'] and row['capex'] > existing_row['capex'])
            ):
                # The new point is dominated by an existing point.
                non_dominated = False
                break
            elif (
                    (row['co2'] <= existing_row['co2'] and row['capex'] <= existing_row['capex']) or
                    (row['co2'] < existing_row['co2'] and row['capex'] < existing_row['capex'])
            ):
                # The new point dominates an existing point.
                to_remove.append(i)

        # Remove dominated solutions.
        if to_remove:
            pareto_front = [i for j, i in enumerate(pareto_front) if j not in to_remove]

        # Add the current row to the Pareto front if it's non-dominated.
        if non_dominated:
            pareto_front.append(row)

    # Create a DataFrame from the Pareto front solutions.
    pareto_front_df = pd.DataFrame(pareto_front)

    # Reset the index of the Pareto front DataFrame for a cleaner result.
    pareto_front_df = pareto_front_df.reset_index(drop=True)

    # Now, 'pareto_front_df' contains the Pareto front solutions.
    return pareto_front_df


def calculate_storage_price():
#Calculate the storage price for the different capacities
    storage_price_unit = 2000 #$/kWh
    storage_price = (storage_counter*storage_step+storage_min)*storage_price_unit #$/kWh #calculate_storage_price()
    #storage_cost = calculate_storage_cap(i)*storage_price_unit
    return storage_price




def run_loop(radiation_data):
    # runs a loop in which the model is run several times with different storage and photovoltaic panel sizes

    results = pd.DataFrame(columns=['Storage (kWh)', 'PV size (qm)',
                                    'capex', 'co2', 'opex', 'LCOE'])
    
    storage_counter = 0
    run_counter = 0
    storage_cost = (storage_counter*storage_step+storage_min)*2000 #$/kWh #calculate_storage_price()
    
    if enable_mg_plots:
        plot_str = 'Yes (Caution! Plots are generated each run!)'
    else:
        plot_str = 'No'
    print(f'\n\n---Running Loop---'
          f'\n\nParameters:'
          f'\n\nMinimum storage capacity:   {storage_min} kWh  Maximum storage capacity    {storage_max} kWh  '
          f'Step size:      {storage_step} kWh'
          f'\nMinimum PV size:              {pv_min} qm       Maximum PV size:            {pv_max} qm        '
          f'Step size:      {pv_step} qm'
          f'\n\nStart day:    {start_day}'
          f'\nDays to run:  {days_to_run}'
          f'\n\nResults will be saved as:  {results_filename}'
          f'\n\nFirst run starts in {pause_time} seconds...'
          f'\n----------------------------------------\n')
    time.sleep(pause_time)
    # run mtress with different storage sizes and PV sizes
    # counters for later use:

    for i in range(storage_min, storage_max + 1, storage_step):
        calculate_storage_cap(i)
        for j in range(pv_min, pv_max + 1, pv_step):
            calulate_pv_generation(radiation_data, j)

            print(f'\nRun #{run_counter + 1} '
                  f'\n----------------------------------------'
                  f'\nParemeters:\n'
                  f'\nStorage size:             {i} kWh'
                  f'\nPV size:                  {j} qm\n'
                  f'----------------------------------------\n\n')
            meta_model, capex, co2_emissions = opti_el_supply(j, storage_counter, storage_cost)
            print('\n--- Run completed, extracting results ---\n')
            
            # extracts results
            keys = meta_model.energy_system.results['main'].keys()
            dfkeys =  pd.DataFrame(keys)
            dfkeys.to_csv(os.path.join(dir_path, 'model_keys'))
            
            #print (aggreg_flows = meta_model.aggregate_flows.flows) -  for LCOE
                        
            el_demand = extract_result_sequence(meta_model.energy_system.results['main'], 'd_el_local')#,'b_elprod'
            el_supply_from_grid = meta_model.energy_system.results['main'][('b_el_adjacent', 'b_grid_connection_in')]['sequences']['flow']            
            el_prod_PV = meta_model.energy_system.results['main'][('t_pv','b_el_pv')]['sequences']['flow']
            el_batt_disch = meta_model.energy_system.results['main'][('s_battery','b_elprod')]['sequences']['flow']
            
            el_PV_export = meta_model.energy_system.results['main'][('b_el_pv', 'b_elxprt')]['sequences']['flow']            
            el_PV_battery = meta_model.energy_system.results['main'][('b_elprod','s_battery')]['sequences']['flow']            
            el_PV_consumed = meta_model.energy_system.results['main'][('b_elprod','b_eldist')]['sequences']['flow']            
            
            el_export_total = meta_model.energy_system.results['main'][('b_elxprt','b_grid_connection_out')]['sequences']['flow']            

            #Print results to csv file for further analysis
            
           
           
            # save capex in column 1, co2 in column 2 of the dataframe
            results.loc[run_counter, 'Storage (kWh)'] = i
            results.loc[run_counter, 'PV size (qm)'] = j
            results.loc[run_counter, 'capex'] = capex
            results.loc[run_counter, 'co2'] = co2_emissions
            if enable_mg_plots:
                # generating plots

                plt.figure()
                plt.plot(el_demand)
                plt.plot(el_prod_PV)
                plt.plot(el_supply_from_grid) 
                plt.title("Demand supplied")
                plt.legend(labels=["El demand", 'PV prod', 'Electr. from grid'])
                
                plt.figure()
                plt.plot(el_demand)
                plt.plot(el_PV_consumed)
                plt.plot(el_batt_disch) 
                plt.plot(el_supply_from_grid) 
                plt.title("Demand supplied")
                plt.legend(labels=["El demand", 'PV to demand', 'Battery discharge', 'El from grid'])
                
                plt.figure()
                plt.plot(el_prod_PV)
                plt.plot(el_PV_export) 
                plt.plot(el_PV_consumed) 
                plt.plot(el_PV_battery)
                plt.title("PV_Energy flows")
                plt.legend(labels=["PV Production", 'PV Export', 'PV consumed', 'PV Battery'])

                #plt.figure()
                #plt.plot(batt_content)
                #plt.title("Storage")
                #plt.legend(labels=['Inflow', 'Outflow'])
                plt.show()
                
            run_counter = run_counter + 1
        storage_counter = storage_counter + 1

    # creates 2 plots of the same results, one with  colormap based on PV panel size in qm and one
    # with colormap based on storage size in liters

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(results['capex'], results['co2'], c=results['Storage (kWh)'],
                cmap='jet', marker='+')
    pareto_front = calculate_pareto(results)
    plt.plot(pareto_front['capex'], pareto_front['co2'], marker='+', color='green', alpha = 0.1)
    cbar = plt.colorbar()
    cbar.set_label('Storage Size (kWh)')

    plt.title("CO2 Emissions vs. Annual Total Costs (Colormap by Storage size)")
    plt.xlabel("Costs in €/year")
    plt.ylabel("CO2 emissions in Kg")

    plt.subplot(1, 2, 2)
    plt.scatter(results['capex'], results['co2'], c=results['PV size (qm)'],
                cmap='jet', marker='+')

    plt.plot(pareto_front['capex'], pareto_front['co2'], marker='+', color='green', alpha = 0.1)
    plt.title("CO2 emissions vs. Annual Total Costs (Colormap by PV panel size in qm)")
    cbar = plt.colorbar()
    cbar.set_label('PV Panel Size (qm)')
    plt.xlabel("Costs in €/year")
    plt.ylabel("CO2 emissions in Kg")
    plt.show()

    # save results in file
    results.to_csv(os.path.join(dir_path, results_filename))

    print('\n -- All runs completed successfully --')


def opti_el_supply(j,storage_counter, storage_cost): 
    # runs the model with the defined parameters, calculating the costs and extracting the co2 emissions. Displays
    # information about this after each run, als well as information about self-sufficiency and own consumption
    lifetime_PV = 25 # lifetime of PV modules in years
    lifetime_battery = 10 # lifetime of batteries modules in years
    meta_model = run_mtress(parameters=yaml_file_name, time_range=(first_time_step, last_time_step), solver = "cbc")
    capex = np.around((cost_per_qm * j)/lifetime_PV + (meta_model.operational_costs()) + (storage_cost)*(lifetime_PV/lifetime_battery)/lifetime_PV, 2) # this are total costs per year 
    co2_emissions = meta_model.co2_emission().sum()
    opex = meta_model.operational_costs()
    capex_only= np.around((cost_per_qm * j)/lifetime_PV + (storage_cost)*(lifetime_PV/lifetime_battery)/lifetime_PV, 2) 
    
    
    print('\n')
    print('KPIs')
    print("Total costs {:.2f} €".format(capex))
    print("CO2 Emission: {:.0f} Kg".format(co2_emissions))
    print("Own Consumption: {:.1f} %".format(meta_model.own_consumption() * 100))
    print("Self Sufficiency: {:.1f} %".format(meta_model.self_sufficiency() * 100))
    print("OPEX {:.2f} €".format(opex))
    print("Capex {:.2f} €".format(capex_only))
    
    print('\n')

    print("")
    
    el_demand = meta_model.aggregate_flows(meta_model.demand_el_flows).sum()
    LCOE = capex/(meta_model.aggregate_flows(meta_model.demand_el_flows).sum())
    print("Electricity demand: {:6.3f}".format(el_demand))
    print("LCOE {:.2f} €/kWh".format(LCOE))

    return meta_model, capex, co2_emissions


def extract_result_sequence(results, label, resample=None):
    """
    :param results:
    :param label:
    :param resample: resampling frequency identifier (e.g. 'D')
    :return:
    """
    sequences = views.node(results, label)['sequences']
    if resample is not None:
        sequences = sequences.resample(resample).mean()
    return sequences


if __name__ == '__main__':
    # init
    # read radiation data for pv generation
    radiation_data = calculate_radiation_data()
    # read input data, keep all in MWh - only valid if no thermal flows modeled & aggregates data to calculate power demand
    read_input_data()
    #
    # run loop
    run_loop(radiation_data)
