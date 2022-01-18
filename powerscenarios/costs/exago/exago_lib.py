# checkmark.py
# THe following file contains the class for costing scenarios using the
# checkmark model

import numpy as np
import pandas as pd
import os, sys, time
import re
import subprocess
import warnings
from os import path
import shutil
from mpi4py import MPI

from powerscenarios.costs.abstract_fidelity import AbstractCostingFidelity
from powerscenarios.costs.exago.matpowerhandler import MatpowerHandler
from exago.opflow import OPFLOW
import os
from pathlib import Path
from exago import config

class ExaGO_Lib(AbstractCostingFidelity):
    """
    This class contains the wrapper for calling OPFLOW from within Powerscenaios
    """
    def __init__(self,
                 grid_name,
                 n_scenarios, # Number of scenarios we actually want in our final csv file
                 n_periods,
                 loss_of_load_cost,
                 spilled_wind_cost,
                 scenarios_df,
                 p_bin,
                 total_power_t0,
                 WTK_DATA_PRECISION=6,
                 nscen_priced=1,
                 mpi_comm=None,
                 ps_gens_df=None,
                 ):

        AbstractCostingFidelity.__init__(self,
                                         n_scenarios,
                                         n_periods,
                                         loss_of_load_cost,
                                         spilled_wind_cost,
                                         scenarios_df,
                                         p_bin,
                                         total_power_t0,
                                         WTK_DATA_PRECISION=6)

        self.grid_name = grid_name # "ACTIVSg200"
        self.opflow_options_dict = {'opflow_solver' : 'IPOPT',
                                    'matpower_file' : "{0}.m".format(self.grid_name),
                                   }
        self.sopflow_options_dict = {'nscenarios' : nscen_priced, # This needs to be a string as the function runs a bash command
                                    }

        self.comm = mpi_comm # MPI communicator
        self._create_ego_object(ps_gens_df=ps_gens_df)

        return

    def _create_ego_object(self,
                           network_file=None,
                           ps_gens_df=None,
                           ):

        # Data files for creating the file based exago object
        base_dir = Path(__file__).parents[3]
        grid_data_dir = os.path.join(base_dir,"data","grid-data")
        # print("base_dir: {}".format(base_dir))
        # print("grid_data_dir: {}".format(grid_data_dir))

        network_file = os.path.join(grid_data_dir,"{0}/case_{0}.m".format(self.grid_name))
        # grid_aux_file = os.path.join(grid_data_dir,"{0}/{0}.aux".format(self.grid_name))
        # print("network_file: {}".format(network_file))
        # print("grid_aux_file: {}".format(grid_aux_file))

        network_file = os.path.join(grid_data_dir,"{0}/case_{0}.m".format(self.grid_name))

        self.ego = ExaGO_Python(network_file,
                                self.grid_name,
                                # real_load_file,
                                # reactive_load_file,
                                comm=self.comm,
                                ps_gens_df=ps_gens_df,
                                )
        # if self.ego.comm.Get_rank() == 0:
        #     self.ego._cleanup() # Lets clean up the file based implementation.

        return


    def compute_scenario_cost(self,
                              actuals_df,
                              binned_scenarios_df,
                              start_time,
                              random_seed=np.random.randint(2 ** 31 - 1),
                              save_opf_results=False,
                              ):

        stop_time = start_time # For now
        my_mpi_rank = self.ego.comm.Get_rank()
        comm_size = self.ego.comm.Get_size()

        # Create Persistence Forecast
        persistence_wind_fcst_df = actuals_df.loc[start_time:stop_time,:].copy().drop(columns=["TotalPower"])
        persistence_wind_fcst_df.index = persistence_wind_fcst_df.index + pd.Timedelta(minutes=5.0)
        # display(persistence_wind_fcst_df)
        # There are some dummy/unused variables currently being used in the funciton
        # calls that we will set to None.
        pv_fcst_df = None
        prev_set_points = None
        n_periods = 1
        step = 5.0
        (base_cost, set_points) = self.ego.base_cost(start_time,
                                           pv_fcst_df, # Currently unused
                                           persistence_wind_fcst_df,
                                           prev_set_points, # Currently unused
                                           n_periods, # Currently unused
                                           step, # Currently unused
                                           self.opflow_options_dict,
                                           system="Mac"
                                           )
        if my_mpi_rank == 0:
            print("\nbase_cost = ", base_cost, "\n")
        # Check if base cost is the same on all ranks
        self.ego.comm.Barrier()
        base_cost_arr = self.ego.comm.gather(base_cost, root=0)
        if my_mpi_rank == 0:
            base_cost_arr = np.asarray(base_cost_arr)
            assert abs(base_cost_arr - base_cost).all() < 1.e-6
        self.ego.comm.Barrier()

        # Turn deviations into scenarios
        wind_scen_df = binned_scenarios_df + persistence_wind_fcst_df.loc[:,binned_scenarios_df.columns].values
        for wgen in wind_scen_df.columns:
            # Enforce Pmax on wind scenarios
            wgen_max = self.ego.wind_max.loc[wgen]
            idx = wind_scen_df.loc[:,wgen] > wgen_max
            wind_scen_df.loc[idx,wgen] = wgen_max
            # Enforce Pmin on wind scenarios
            idx = wind_scen_df.loc[:,wgen] < 0.0
            wind_scen_df.loc[idx,wgen] = 0.0

        self.ego.comm.Barrier()
        # if my_mpi_rank == 0:
        #     print('Available Scenarios = ', wind_scen_df.shape[0], ", Requested Scenarios = ", int(self.sopflow_options_dict['nscenarios']))

        assert wind_scen_df.shape[0] >= int(self.sopflow_options_dict['nscenarios'])
        nscen_global = int(self.sopflow_options_dict['nscenarios']) # wind_scen_df.shape[0]

        nscen_local_arr = np.zeros(comm_size, dtype=int)
        quotient, remainder = divmod(nscen_global, comm_size)
        nscen_local_arr[:] = quotient
        nscen_local_arr[0:remainder] += 1 # Divide the remainder evenly across the first n_remainder ranks
        # Sanity Check of MPI
        assert nscen_local_arr.sum() == nscen_global, "Scenarios were not properly divided across the MPI ranks"

        # Now get the starting and stopping location of a given rank
        if my_mpi_rank == 0:
            idx_offset = 0
        else:
            idx_offset = nscen_local_arr[0:my_mpi_rank].sum() # np.dot(np.arange(my_mpi_rank)+1, nscen_local_arr[0:my_mpi_rank])

        q_cost_local = np.zeros(nscen_local_arr[my_mpi_rank])
        q_sts_local = np.zeros(nscen_local_arr[my_mpi_rank], dtype=bool)
        # local_opf_scen_dict = {}
        # print("my_mpi_rank = ", my_mpi_rank, ", idx_offset = ", idx_offset)
        # time_offset = idx_offset*step
        # ts = wind_scen_df.index[0] + pd.Timedelta(minutes=time_offset)
        for i in range(nscen_local_arr[my_mpi_rank]):
            ts = wind_scen_df.index[idx_offset + i] # Will not work for MPI
            # local_opf_scen_dict[i] = OPFLOW()
            # opf_object = local_opf_scen_dict[i] # For convenience as of now
            opf_object = OPFLOW()
            opf_object.dont_finalize()
            opf_object.read_mat_power_data(self.ego.network_file)
            opf_object.set_include_loadloss(False)
            # opf_object.set_loadloss_penalty(833)
            opf_object.set_include_powerimbalance(True)
            # opf_object.set_powerimbalance_penalty(833)
            opf_object.set_model('POWER_BALANCE_POLAR')
            opf_object.set_solver('IPOPT')
            opf_object.set_initialization('ACPF')
            opf_object.set_genbusvoltage('VARIABLE_WITHIN_BOUNDS')
            opf_object.setup_ps()

            # Constrain the thermal generation for the second stage
            self.ego._fix_non_wind(opf_object, set_points)

            # The following two lines are copied directly from base_cost for now.
            # Revisit them again
            wf_df = wind_scen_df.loc[ts:ts, :]
            self.ego._set_wind_new(opf_object,
                                   wf_df,
                                   self.ego.imat.get_table('gen'),
                                   self.ego.gen_type)

            # Run the opflow realization
            opf_object.solve()
            opf_object.solution_to_ps()

            sts = opf_object.convergence_status()
            q_sts_local[i] = sts
            q_cost_local[i] = opf_object.objective_function
            # if sts == False or q_cost_local[i] < base_cost or save_opf_results:
            if (sts == False or
                save_opf_results or
                opf_object.objective_function + 10.0 < base_cost
                ):
                ts_str = str(ts).replace(' ', 'T').replace(':','_').split('+')[0]
                soln_file = 'ego_opflow_{}.m'.format(ts_str)
                opf_object.save_solution('MATPOWER', soln_file)


            # Increment the time-stamp so as to price other scenarios
            # ts += pd.Timedelta(minutes=step)

        q_cost_local -= base_cost

        # if my_mpi_rank == 0:
        #     q_cost_global = np.empty(nscen_global, dtype=float)
        # else:
        #     q_cost_global = None
        #     cost_n = None
        q_cost_global = np.empty(nscen_global, dtype=float)
        q_sts_global = np.empty(nscen_global, dtype=bool)

        self.ego.comm.Barrier()
        # self.ego.comm.Gatherv(sendbuf=q_cost_local,
        #                   recvbuf=(q_cost_global, nscen_local_arr),
        #                   root=0)
        self.ego.comm.Allgatherv(sendbuf=q_cost_local,
                                 recvbuf=(q_cost_global, nscen_local_arr))
        cost_n = pd.Series(index=wind_scen_df.index, data=q_cost_global)

        self.ego.comm.Allgatherv(sendbuf=q_sts_local,
                                 recvbuf=(q_sts_global, nscen_local_arr))
        sts_n = pd.Series(index=wind_scen_df.index, data=q_sts_global)
        if my_mpi_rank == 0:
            idx = sts_n == False
            if np.sum(idx) > 0:
                unconv_file = "ExaGO_unconverged_{}.csv".format(self.grid_name)
                sts_n[idx].index.to_series().to_csv(unconv_file)
                print("ExaGO OPFLOW did not converge for {} timestamps. Timestamps have been printed to the file: {}".format(np.sum(idx), unconv_file))

            idx = cost_n < 0.0
            if np.sum(idx) > 0:
                # print("Negative prices found for the following timestamps:\n",
                #       cost_n[idx].index)
                print("Negative prices have been found for {} timestamps. Largest magnitude is {}".format(np.sum(idx), cost_n[idx].abs().max()))

        # for i in range(comm_size):
        #     if my_mpi_rank == i:
        #         # Zip the numpy array into the timeseries
        #         # cost_n = pd.Series(index=wind_scen_df.index, data=q_cost_global)
        #         print("rank = ", my_mpi_rank)
        #         print("q_cost_global = ", repr(q_cost_global))
        #         # np.savetxt("cost_lib.txt", q_cost_global)
        #         # print("q_cost_global max = ", np.amax(q_cost_global))
        #         # print("q_cost_global min = ", np.amin(q_cost_global))
        #     self.ego.comm.Barrier()

        return cost_n

###############################################

class ExaGO_Python:

    def __init__(self,
                 network_file,
                 grid_name,
                 load_dir=None,
                 real_load_file=None,
                 reactive_load_file=None,
                 year=2020,
                 comm=None,
                 ps_gens_df=None,
                 ):

        # print("Loading network from file: ", network_file)

        # MPI specifics
        if comm == None:
            print("No MPI comm provided, establishing a new one")
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm
        my_mpi_rank = self.comm.Get_rank()

        # self.exe_path = exe_path
        self.grid_name = grid_name

        # self.network_file = network_file
        self.exago_ignore = -1.e6

        self.imat = MatpowerHandler(network_file)

        # Overwrite max and min gens if provided
        if ps_gens_df is not None:
            gen_df = self.imat.get_table('gen')
            gen_df['Pmax'] = ps_gens_df.loc[:,'GenMWMax']
            gen_df['Pmin'] = ps_gens_df.loc[:,'GenMWMin']

        self.gen_df_org = self.imat.get_table('gen').copy()
        self.bus_df_org = self.imat.get_table('bus').copy()
        self.gids = self._assign_gen_ids(self.gen_df_org)

        # # print("Reading in load data...")
        # # Read in the load dataframes
        # if real_load_file is None:
        #     self.p_load_df = None
        # else:
        #     p_df = pd.read_csv(real_load_file, index_col=0, parse_dates=True)
        #     p_df.index = p_df.index.map(lambda t: t.replace(year=year))
        #     self.p_load_df = p_df

        # if reactive_load_file is None:
        #     self.q_load_df = None
        #     # raise ValueError("The reactive load file has not been specified.")
        # else:
        #     q_df = pd.read_csv(reactive_load_file, index_col=0, parse_dates=True)
        #     q_df.index = q_df.index.map(lambda t: t.replace(year=year))
        #     self.q_load_df = q_df

        gen_type = self.imat.static['genfuel'].split('\n')[1:-2]
        assert len(gen_type) == self.gen_df_org.shape[0]
        self.gen_type = pd.Series(map(lambda s: s.strip(' \t\r\n\';)'), gen_type))

        idx = self._wind_gens(self.gen_type)

        buses = self.gen_df_org.loc[idx,'bus']
        gen_id = pd.Series(data=1, index=buses.index)
        for bus in set(buses):
            bidx = buses == bus
            gen_id.loc[bidx] = range(1,sum(bidx)+1)

        if ps_gens_df is None:
            pmax = self.gen_df_org.loc[idx,'Pmax']
            pmax.index = (buses.apply(str)
                          + '_Wind_'
                          + gen_id.apply(str))
            self.wind_max = pmax
        else:
            idx = self._wind_gens(self.gen_type)
            wind_gen_max = ps_gens_df.loc[idx,'GenMWMax']
            self.wind_max = wind_gen_max
            self.wind_max.index = ps_gens_df.loc[idx,'GenUID']

        matpower_file = os.path.join('EXAGO_LIB_case_{}.m'.format(self.grid_name))
        self.imat.write_matpower_file(matpower_file)
        self.network_file = matpower_file

        return


    def _non_wind_gens(self,
                       gen_type):
        return np.logical_not(self._wind_gens(gen_type))

    def _wind_gens(self,
                   gen_type):
        return (gen_type == 'wind')


    def _restore_org_gen_table(self):
        gen_df = self.imat.get_table('gen')
        for col in ['Pmax', 'Pmin', 'Pg', 'Qmax', 'Qmin', 'Qg']:
            gen_df.loc[:,col] = self.gen_df_org.loc[:,col]
        return

    # def _set_load(self,
    #               p_df,
    #               q_df,
    #               bus_df):
    #     # Bus loads not in the given dataframes are assumed to be zero
    #     bus_df.loc[:,'Pd'] = 0.0
    #     for bus in pd.to_numeric(p_df.columns):
    #         idx = (bus_df.loc[:,'bus_i'] == bus)
    #         bus_df.loc[idx,'Pd'] = p_df.loc[p_df.index[0],str(bus)]
    #     bus_df.loc[:,'Qd'] = 0.0
    #     for bus in pd.to_numeric(q_df.columns):
    #         idx = (bus_df.loc[:,'bus_i'] == bus)
    #         bus_df.loc[idx,'Qd'] = q_df.loc[q_df.index[0],str(bus)]
    #     return

    # def _scale_load(self, bus_df, scaling_factor):
    #     bus_df.loc[:,'Pd'] = scaling_factor*bus_df.loc[:,'Pd']
    #     bus_df.loc[:,'Qd'] = scaling_factor*bus_df.loc[:,'Qd']
    #     return

    def _assign_gen_ids(self, gen_df):
            gbuses = gen_df.loc[:,'bus']
            gids = ['1 '] * gbuses.size

            if gbuses.size > gbuses.unique().size:
                next_gid = {bus:1 for bus in gbuses}
                for (k,bus) in enumerate(gen_df.loc[:,'bus']):
                    idx = gen_df.loc[:,'bus'] == bus
                    if idx.sum() > 1:
                        gids[k] = "{:<2d}".format(next_gid[bus])
                        next_gid[bus] += 1
                    else:
                        pass

            elif gbuses.size < gbuses.unique().size:
                assert False
            else:
                pass
            return gids

    """
    def _assign_gen_ids(self, gen_df):
        gbuses = gen_df.loc[:,'bus']
        gids = ['1 '] * gbuses.size
        if gbuses.size > gbuses.unique().size:
            for (k,bus) in enumerate(gbuses.unique()):
                idx = gen_df.loc[:,'bus'] == bus
                if idx.sum() > 1:
                    for (gid,gbus) in enumerate(gen_df.loc[idx,'bus']):
                        print("Giving generator at bus {} id {}".format(gbus, gid+1))
                        assert bus == gbus
                        gids[k] = "{:<2d}".format(gid)
                else:
                    pass
        elif gbuses.size < gbuses.unique().size:
            assert False
        else:
            pass
        return gids
    """

    def _set_wind_new(self,
                      opf,
                      w_df,
                      gen_df,
                      gen_type
                      ):

        idx = self._wind_gens(gen_type)
        assert w_df.columns.size == idx.sum()

        for wgen in w_df.columns:
            bus = int(wgen.split('_')[0])
            gen_id = int(wgen.split('_')[2]) # - 1
            widx = (gen_df.loc[:,'bus'] == bus) & (idx)
            if widx.sum() >= 1:
                if gen_id < 10:
                    g_id = str(gen_id) + " "
                else:
                    g_id = str(gen_id)
                opf.ps_set_gen_power_limits(bus,
                                            g_id,
                                            w_df.loc[w_df.index[0], wgen],
                                            0,
                                            self.exago_ignore,
                                            self.exago_ignore
                                            )
            else:
                print("Unable to identify row corresponding to generator {}".format(wgen))

        # self.opf.ps_set_gen_power_limits(3, "1 ", 65, 0, exago_ignore, exago_ignore)
        return

    def _fix_non_wind(self, opflow_object, set_points):
        idx = self._non_wind_gens(self.gen_type) # Bool array of non-wind set points
        og_gen_df = self.imat.get_table('gen')   # Original generatar table from Matpower file
        pg_arr = set_points[0] # Real set-points
        qg_arr = set_points[1] # Reactive set points
        for i in range(og_gen_df.shape[0]):
            if idx[i]:
                bus = int(og_gen_df.loc[i, "bus"])
                opflow_object.ps_set_gen_power_limits(bus, self.gids[i],
                                                      pg_arr[i], pg_arr[i],
                                                      qg_arr[i], qg_arr[i])
        return


    def _extract_set_points(self, opf):
        idx = self._non_wind_gens(self.gen_type) # Bool array of non-wind set points
        og_gen_df = self.imat.get_table('gen')   # Original generatar table from Matpower file
        pg_set = np.zeros(og_gen_df.shape[0]) # real set points array
        qg_set = np.zeros(og_gen_df.shape[0]) # reactive set points array

        for i in range(og_gen_df.shape[0]):
            if idx[i]:
                # We will only collect the set points of the non-wind_generators
                # print("i = ", i, ", self.gids[i] = ", repr(self.gids[i]))
                pg_set[i], qg_set[i] = opf.get_gen_dispatch(int(og_gen_df.loc[i, "bus"]), self.gids[i])
                # print("bus ", int(og_gen_df.loc[i, "bus"]), " pg = ", pg_set[i], " qg = " , qg_set[i])

        return (pg_set, qg_set)


    # def _cleanup(self):
    #     if path.exists("opflowout.m"):
    #         os.remove("opflowout.m")
    #     if path.exists("sopflowout"):
    #         shutil.rmtree("sopflowout")
    #     if path.exists("case_{0}.m".format(self.grid_name)):
    #         os.remove("case_{0}.m".format(self.grid_name))

    def base_cost(self,
                  start_time,
                  pv_fcst_df, # Currently unused
                  wind_fcst_df,
                  prev_set_points, # Currently unused
                  n_periods, # Currently unused
                  step, # Currently unused
                  opflow_options_dict,
                  system="Summit"
                  ):

        # t0 = time.time()

        # stop_time = start_time
        # self._restore_org_gen_table()

        # Create OPFLOW object
        self.opf_base = OPFLOW()
        self.opf_base.dont_finalize()
        self.opf_base.read_mat_power_data(self.network_file)
        # self.opf_base.read_mat_power_data(matpower_file)
        self.opf_base.set_include_loadloss(False)
        self.opf_base.set_include_powerimbalance(False)
        # self.opf_base.set_include_loadloss(True)
        # self.opf_base.set_loadloss_penalty(833)
        # self.opf_base.set_include_powerimbalance(True)
        # self.opf_base.set_powerimbalance_penalty(933)
        self.opf_base.set_model('POWER_BALANCE_POLAR')
        self.opf_base.set_solver('IPOPT')
        self.opf_base.set_initialization('ACPF')
        # self.opf_base.set_genbusvoltage('VARIABLE_WITHIN_BOUNDS')
        self.opf_base.setup_ps()

        wf_df = wind_fcst_df
        self._set_wind_new(self.opf_base, wf_df, self.imat.get_table('gen'), self.gen_type)

        # Python call
        self.opf_base.solve()
        if self.comm.Get_rank() == 0:
            self.opf_base.save_solution('MATPOWER', "base_{0}.m".format(self.grid_name))
        self.opf_base.solution_to_ps()

        sts = self.opf_base.convergence_status()
        if not sts:
            raise Exception("Base case failed to converge. Unable to cost scenarios.")

        # t1 = time.time()

        obj = self.opf_base.objective_function
        set_points = self._extract_set_points(self.opf_base)
        # idx = self._non_wind_gens(self.gen_type)
        # set_points = result.get_table('gen').loc[idx,:]

        # t2 = time.time()

        # total_elapsed = t2 - t0

        # if self.comm.Get_rank() == 0:
        #     print("**** Base Cost Timing ****")
        #     print("Total time in function = {0}".format(total_elapsed))
        #     print("Time in base cost solve = {0}".format(t1 - t0))

        return (obj, set_points)

    """
    def cost_scenarios(self,
                           start_time,
                           pv_fcst_df, # Currently unused
                           wind_fcst_df,
                           wind_dev_df,
                           prev_set_points, # Currently unused
                           opflow_options_dict,
                           sopflow_options_dict,
                           n_periods=1, # Currently unused
                           step=5.0, # Currently unused
                           system="Summit"
                           ):

        my_mpi_rank = self.comm.Get_rank()
        comm_size = self.comm.Get_size()
        if my_mpi_rank == 0:
            # self._cleanup() # We may not need it so candidate for deletion
            pass

        # Lets run the base cost simulation on every rank and scenario binning
        (base_cost, set_points) = self.base_cost_lib(start_time,
                                           pv_fcst_df, # Currently unused
                                           wind_fcst_df,
                                           prev_set_points, # Currently unused
                                           n_periods, # Currently unused
                                           step, # Currently unused
                                           opflow_options_dict,
                                           system=system
                                           )
        # Check if base cost is the same on all ranks
        self.comm.Barrier()
        base_cost_arr = self.comm.gather(base_cost, root=0)
        if my_mpi_rank == 0:
            base_cost_arr = np.asarray(base_cost_arr)
            assert abs(base_cost_arr - base_cost).all() < 1.e-6
        self.comm.Barrier()

        # Turn deviations into scenarios
        stop_time = start_time
        w_scen_df = wind_fcst_df.loc[start_time:stop_time,:]
        wind_scen_df = wind_dev_df + w_scen_df.loc[:,wind_dev_df.columns].values
        for wgen in wind_scen_df.columns:
            # Enforce Pmax on wind scenarios
            wgen_max = self.wind_max.loc[wgen]
            idx = wind_scen_df.loc[:,wgen] > wgen_max
            wind_scen_df.loc[idx,wgen] = wgen_max
            # Enforce Pmin on wind scenarios
            idx = wind_scen_df.loc[:,wgen] < 0.0
            wind_scen_df.loc[idx,wgen] = 0.0

        self.comm.Barrier()
        if my_mpi_rank == 0:
            print('Available Scenarios = ', wind_scen_df.shape[0], ", Requested Scenarios = ", int(sopflow_options_dict['nscenarios']))

        assert wind_scen_df.shape[0] >= int(sopflow_options_dict['nscenarios'])
        nscen_global = int(sopflow_options_dict['nscenarios']) # Total number of scenarios that need to be priced

        nscen_local_arr = np.zeros(comm_size, dtype=int)
        quotient, remainder = divmod(nscen_global, comm_size)
        nscen_local_arr[:] = quotient
        nscen_local_arr[0:remainder] += 1 # Divide the remainder evenly across the first n_remainder ranks
        # Sanity Check of MPI
        assert nscen_local_arr.sum() == nscen_global, "Scenarios were not properly divided across the MPI ranks"

        # Now get the starting and stopping location of a given rank
        if my_mpi_rank == 0:
            idx_offset = 0
        else:
            idx_offset = nscen_local_arr[0:my_mpi_rank].sum() # np.dot(np.arange(my_mpi_rank)+1, nscen_local_arr[0:my_mpi_rank])

        q_cost_local = np.zeros(nscen_local_arr[my_mpi_rank])
        local_opf_scen_dict = {}
        time_offset = idx_offset*step
        ts = wind_scen_df.index[0] + pd.Timedelta(minutes=time_offset)
        for i in range(nscen_local_arr[my_mpi_rank]):
            local_opf_scen_dict[i] = OPFLOW()
            opf_object = local_opf_scen_dict[i] # For convenience as of now
            opf_object.read_mat_power_data(self.network_file)
            opf_object.setup_ps()

            # Constrain the thermal generation for the second stage
            self._fix_non_wind(opf_object, set_points)

            # The following two lines are copied directly from base_cost for now.
            # Revisit them again
            wf_df = wind_scen_df.loc[ts:ts, :]
            self._set_wind_new(opf_object, wf_df, self.imat.get_table('gen'), self.gen_type)

            # Run the opflow realization
            opf_object.solve()
            opf_object.solution_to_ps()

            q_cost_local[i] = opf_object.objective_function

            # Increment the time-stamp so as to price other scenarios
            ts += pd.Timedelta(minutes=step)

        q_cost_local -= base_cost

        if my_mpi_rank == 0:
            q_cost_global = np.empty(nscen_global, dtype=float)
        else:
            q_cost_global = None
        self.comm.Barrier()

        self.comm.Gatherv(sendbuf=q_cost_local,
                          recvbuf=(q_cost_global, nscen_local_arr),
                          root=0)

        self.comm.Barrier()

        return q_cost_global
    """
