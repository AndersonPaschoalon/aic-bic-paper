#!/usr/bin/python3.5
"""
This script generate the AIC/BIC CSV file and calc its relative difference
"""
import sys
import numpy
from Utils.Terminal import Terminal as term
from Utils.Csv import Csv
from Utils.Matrix import Matrix

__author__ = 'Anderson Paschoalon'
__copyright__ = ""
__credits__ = ["Anderson Paschoalon", "Christian Esteve Rothenberg"]
__license__ = "MIT"
__version__ = ""
__maintainer__ = "Anderson Paschoalon"
__email__ = "anderson.paschoalon@gmail.com"
__status__ = "Production"

#PATH_DIR = str(sys.argv[1])
PATH_DIR = './data/'

def table2csv(filename):
    """
    Convert a ASCII table to CSV format
    :param filename:
    :return:
    """
    cmd = "cat {0} |sed 's/\(^|\)\|\(^+\(-\|+\)*\)\|\(|$\)//g' |sed 's/Function/#Function/g' |sed 's/|/,/g'  |sed 's/[[:blank:]]//g' |sed '/^$/d' | awk 'BEGIN{{print(\"# AIC and BIC values\")}}{{print $0}}' > {0}.csv"
    cmd = cmd.format(filename)
    term.command(cmd=cmd, color="green")


########################################################################################################################
# Main
########################################################################################################################

if __name__ == "__main__":
    aic_bic_dat = PATH_DIR + "/Aic-Bic.dat"
    aic_bic_csv = PATH_DIR + "/Aic-Bic.dat.csv"
    table2csv(PATH_DIR + "/Aic-Bic.dat")
    csv_data = Csv.load_csv_str(aic_bic_csv)
    labels = Matrix.column(csv_data, 0)
    aic_values = [float(val) for val in Matrix.column(csv_data, 1)]
    bic_values = [float(val) for val in Matrix.column(csv_data, 2)]
    relative_diff = [ 100*abs(aic_values[i] - bic_values[i])/((abs(aic_values[i]) + abs(bic_values[i]))/2) for i in range(0, len(bic_values))]
    aic_bic_relative_diff = numpy.column_stack([labels, relative_diff])
    print("\nStochastic Functions: " + str(labels))
    print("aic_values: " + str(aic_values))
    print("bic_values: " + str(bic_values))
    print("\nAIC/BIC relative differences(%): \n" + str(aic_bic_relative_diff))
    numpy.savetxt(PATH_DIR + "/aic_bic_relative_difference.csv", aic_bic_relative_diff, delimiter=", ", fmt='%s',
                  header='StochasticModels, AIC/BIC_relative_difference(%)')





