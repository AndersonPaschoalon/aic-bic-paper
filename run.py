#!/usr/bin/python3.5
import os
import sys
import time
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from Cd.Cd import Cd

#PCAP_FILE = "../../Pcap/wireshark-wiki_http.pcap"
#SIMULATION_NAME = "wombat"


def main(args):
    pcap  = args[0]
    sym_name = args[1]
    # input arguments
    pcap_file = os.path.dirname(os.path.abspath(__file__)) + '/' + pcap
    # making sure the program is being executed in the source location, so it can be executed from anyware
    cd = Cd(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = 'plots/' + sym_name
    # run simulations
    run_simulations(pcap_file, sym_name, plots_dir, cd)
    # plot data
    #plot_data(pcap_file, sym_name, plots_dir)

def dataprocessor_help():
    print('./run.py <pcap_file>  <simulation_name>')
    print('\tEg.: ./run.py ../pcaps/wireshark-wiki_http.pcap  wombat')

def print_header(title):
    print('')
    print('###############################################################################')
    print('# ' + title)
    print('###############################################################################')


def run_simulations(pcap_file, sym_name, plots_dir, cd):
    print_header("Simulations for:" + sym_name + " using pcap:" + pcap_file)
    # clean sim dir or create if it does not exist
    cd.cd('./dataProcessor/')
    os.system('mkdir -p figures')
    os.system('mkdir -p data')
    os.system('rm -rf data/*')
    # filter inter-pacekt times
    os.system('./pcap-filter.sh --time-delta ' + pcap_file + ' ' + sym_name)
    # execute dataProcessor prototype
    os.system('./dataProcessor.m ' + sym_name + ' |tee data/dataProcessorStdOut.log')
    # calc cost function of each simulation
    os.system('./calcCostFunction.py ' + 'data/')
    # back to working directory
    cd.back()
    # creating plots dir, clean if already exist
    os.system('rm -rf ' + plots_dir)
    os.system('mkdir -p ' + plots_dir)
    os.system('mv dataProcessor/data/* ' + plots_dir)
    str_about = 'Simulation:' + sym_name + " using pcap:" + pcap_file
    str_date = '@ ' + str(time.localtime().tm_mday) + '/' + str(time.localtime().tm_mon) + '/' + str(
        time.localtime().tm_year) + '-' + str(time.localtime().tm_hour) + ':' + str(
        time.localtime().tm_min) + ':' + str(time.localtime().tm_sec)
    os.system('echo \"' + str_about + '\" >>' + plots_dir + '/about.log')
    os.system('echo \"' + str_date + '\" >>' + plots_dir + '/about.log')





if __name__ == "__main__":
    if (len(sys.argv) == 1) or (len(sys.argv) == 2):
        dataprocessor_help()
    else:
        main(sys.argv[1:])

