#!/usr/bin/python3
# deps install
# pip3 install matplotlib
# import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
# import matplotlib
import sys
import collections
# from Utils.Cd import Cd
from Utils.Terminal import Terminal as term


########################################################################################################################
# Utils
########################################################################################################################

def signedlog(values):
    """

    :param values:
    :return:
    """
    return [i / abs(i) * math.log(abs(i)) for i in values]

def prepare_aic_bic_csv(list_aib_bic_files):
    """

    :param list_aib_bic_files:
    :return:
    """
    for tab_file in list_aib_bic_files:
        table2csv(tab_file)


def table2csv(filename):
    """
    Convert a ASCII table to CSV format
    :param filename:
    :return:
    """
    cmd = "cat {0} |sed 's/\(^|\)\|\(^+\(-\|+\)*\)\|\(|$\)//g' |sed 's/Function/#Function/g' |sed 's/|/,/g'  |sed 's/[[:blank:]]//g' |sed '/^$/d' | awk 'BEGIN{{print(\"# AIC and BIC values\")}}{{print $0}}' > {0}.csv"
    cmd = cmd.format(filename)
    term.command(cmd=cmd, color="green")


def load_csv(datafile=''):
    """
    Load float CSV file into a matrix.
    mtr_float = load_csv(datafile='file.csv')
    :param datafile: CSV file to be loaded
    :return: matrix with the CSV file data
    """
    try:
        with open(datafile) as f:
            lines = (line for line in f if not line.startswith('#'))
            csv_matrix = np.loadtxt(lines, delimiter=',')
        return csv_matrix
    except:
        term.print_color(color="red", data="File {" + datafile + "} not found.")
        sys.exit("File not found")



def load_csv_str(datafile=''):
    """
    Load a String CSV file into a matrix
    :param datafile:
    :return: string matrix with CSV data
    mtr_str = load_csv_str(datafile='file.csv')
    """
    ifile = ""
    try:
        ifile = open(datafile, "rU")
    except:
        term.print_color(color="red", data="File {" + datafile + "} not found.")
        sys.exit("File not found")
    reader = csv.reader(ifile, delimiter=",")
    rownum = 0
    a = []
    for row in reader:
        if len(row) == 0 or row[0][0] == '#':
            continue
        for index in range(0, len(row) - 1):
            row[index] = row[index].strip()
        a.append(row)
        rownum += 1
    ifile.close()
    return a


def column(matrix, i):
    """
    Returns a column of a two dimensional matrix
    mtr_col = column(mtr, 2):
    :param matrix: matrix
    :param i: column index
    :return: vector
    """
    return [row[i] for row in matrix]


def order_matrix(mtr, n_column):
    """
    Order the matrix according to the column n
    m_ordered = order_matrix(m, 1)
    :param mtr:
    :param n_column:
    :return:
    """
    mtr = sorted(mtr, key=lambda mtr: float(mtr[n_column]))
    return mtr


def test_order_matrix():
    """
    #
    :return:
    """
    m = [['abacaxi', '3', '5', '6', '7', '7'], ['banana', '0', '4', '5', '6', '7'],
         ['caqui', '1', '3', '4', '5', '6'], ['damasco', '7', '14', '15', '16', '17'],
         ['damasco', '7', '14', '15', '16', '17'], ['caqui', '7', '14', '15', '16', '17'],
         ['figo', '2', '99', '98', '97', '96'], ['goiaba', '9', '10', '11', '12', '13']]
    m_ordered = order_matrix(m, 1)
    print(m_ordered)


def order_matrix_str(mtr, n_column):
    """
    Order the matrix according to the column n
    m_ordered = order_matrix(m, 1)
    :param mtr:
    :param n_column:
    :return:
    """
    col_str = sorted(column(mtr, n_column))
    mtr_out = []
    for i in range(0, len(col_str)):
        for j in range(0, len(col_str)):
            if mtr[j][n_column] == col_str[i]:
                mtr_out.append(mtr[j])
    return mtr_out


def test_order_matrix_str():
    """
    #
    :return:
    """
    mtr = [['pera', '1', '2', '3'], ['uva', '3', '4', '5'],
           ['abacaxi', '7', '6', '5'], ['banana', 'd', 'f', 'g']]
    mtr = order_matrix_str(mtr, 0)
    print(mtr)


def get_mtr_position(mtr, model):
    """
    Return the position of the model
    :param mtr:
    :param model:
    :return:
    """
    pos = -1
    for i in range(0, len(mtr)):
        if model == mtr[i][0]:
            pos = i
            break
    return pos


def test_get_mtr_position():
    """
    #
    :return:
    """
    m = [['banana', '0', '4', '5', '6', '7'],
         ['caqui', '1', '3', '4', '5', '6'],
         ['figo', '2', '99', '98', '97', '96'],
         ['abacaxi', '3', '5', '6', '7', '7'],
         ['damasco', '7', '14', '15', '16', '17'],
         ['goiaba', '9', '10', '11', '12', '13']]
    pos = get_mtr_position(m, 'damasco')
    print('pos=' + str(pos))


def calc_relative_position_rank_diff(mtr_costfunction, mtr_aicbic):
    """
    #
    :param mtr_costfunction:
    :param mtr_aicbic:
    :return:
    """
    mtr_costfunction = order_matrix(mtr_costfunction, 1)
    mtr_aicbic = order_matrix(mtr_aicbic, 1)
    # print("    Cost Function=" + str(mtr_costfunction))
    # print("    Aic/BIC="+ str(mtr_aicbic))
    vet_relative_rank_diff = []
    for i in range(0, len(mtr_costfunction)):
        model = mtr_costfunction[i][0]
        pos_costfunction = get_mtr_position(mtr_costfunction, model)
        pos_aicbic = get_mtr_position(mtr_aicbic, model)
        vet_relative_rank_diff.append(pos_costfunction - pos_aicbic)
        # print("model:"+model + ", pos_costfunction:"+str(pos_costfunction)+ ", pos_aicbic:"+str(pos_aicbic))
    return vet_relative_rank_diff


def test_calc_relative_position_rank_diff():
    """
    #
    :return:
    """
    # banana, caqui, figo, abacaxi, damasco, goiaba
    m1 = [['banana', '0', '4', '5', '6', '7'],
          ['caqui', '1', '3', '4', '5', '6'],
          ['figo', '2', '99', '98', '97', '96'],
          ['abacaxi', '3', '5', '6', '7', '7'],
          ['damasco', '7', '14', '15', '16', '17'],
          ['goiaba', '9', '10', '11', '12', '13']]
    # banana, figo, caqui, goiaba, abacaxi, damasco
    m2 = [['figo', '0', '4', '5', '6', '7'], ['caqui', '3', '3', '4', '5', '6'],
          ['banana', '-2', '99', '98', '97', '96'],
          ['abacaxi', '5', '5', '6', '7', '7'],
          ['damasco', '7', '14', '15', '16', '17'],
          ['goiaba', '4', '10', '11', '12', '13']]
    # expected : 0, -1, 1, -1, -1, 2
    vet_relative_rank_diff = calc_relative_position_rank_diff(m1, m2)
    print(vet_relative_rank_diff)


def errorbar_helper(ax, xdata, ydata, yerror, param_dict, legend=True):
    """
    #
    :param ax: AxesSubplot object
    :param xdata: x data
    :param ydata: y data
    :param yerror: vertical errorbar
    :param param_dict: list of plot parameters
    :param legend:
    :return:
    """
    out = ax.errorbar(xdata, ydata, yerror, **param_dict)
    if legend: ax.legend()
    return out


def saver_helper(figure_object, file_name="default"):
    """
    Helper for saving figure in many formats
    :param figure_object: object fig
    :param file_name: file name to be saved
    :return: void
    """
    figure_object.savefig(fname=file_name + '.pdf')
    # figure_object.savefig(fname=file_name+'.svg')
    figure_object.savefig(fname=file_name + '.png')
    figure_object.savefig(fname=file_name + '.eps')


def plt_free():
    """
    """
    plt.cla()
    plt.clf()
    plt.close()


def print_info(title="title", location="path-file"):
    """

    :param title:
    :param location:
    :return:
    """
    print("Plotting `{0}` > `{1}`".format(title, location))


########################################################################################################################
# Plot functions
########################################################################################################################

def plot_cdf_fitting(plot_dir, fitting_data, original_datafile, plot_title, plot_file):
    """
    #
    :param plot_dir:
    :param fitting_data:
    :param original_datafile:
    :param plot_title:
    :param plot_file:
    :return:
    """
    # load data
    original_data = load_csv(datafile=plot_dir + original_datafile)
    fitting_data = load_csv(datafile=plot_dir + fitting_data)
    ox = column(original_data, 0)
    oy = column(original_data, 1)
    fx = column(fitting_data, 0)
    fy = column(fitting_data, 1)
    olabel = "empirical"
    flabel = "approximation"
    xlabel = "Inter packet time (s)"
    ylabel = "CDF function"
    # plotting
    print_info(title=plot_file, location=plot_dir)
    fig1, ax1 = plt.subplots()
    ax1.plot(ox, oy, 'r-', label=olabel, linewidth=2)
    ax1.plot(fx, fy, '-.', color="darkblue", label=flabel, linewidth=2.5)
    ax1.legend(loc='lower right')
    # ax1.set_aspect(aspect=1.50)
    plt.xlim([-0.1, 10])
    plt.ylim([0, 1.01])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(color='black', linestyle=':')
    plt.title(plot_title)
    saver_helper(fig1, file_name=plot_dir + "Linear - " + plot_file)
    fig2, ax2 = plt.subplots()
    ax2.plot(ox, oy, 'r-', label=olabel, linewidth=2)
    ax2.plot(fx, fy, '-.', color="darkblue", label=flabel, linewidth=2.5)
    ax2.legend(loc='upper left')
    # ax2.set_aspect(aspect=1.50)
    plt.semilogx()
    plt.ylim([0, 1.01])
    plt.grid(color='black', linestyle=':')
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    saver_helper(fig2, file_name=plot_dir + "Log - " + plot_file)
    plt_free()


def plot_linear_regression(plot_dir, datafile, plot_title, plot_file):
    """
    #
    :param plot_dir:
    :param datafile:
    :param plot_title:
    :param plot_file:
    :return:
    """
    # load data
    original_data = load_csv(datafile=plot_dir + datafile)
    fitting_data = load_csv(datafile=plot_dir + datafile)
    lx = column(original_data, 0)
    ly = column(original_data, 1)
    ax = column(fitting_data, 2)
    ay = column(fitting_data, 3)
    llabel = "linearized data"
    alabel = "linear approximation"
    xlabel = "interPacketTime (s)"
    ylabel = "F(interPacketTime)"
    # plotting
    print_info(title=datafile, location=plot_dir)
    fig1, ax1 = plt.subplots()
    ax1.plot(lx, ly, 'x', color="darkblue", label=llabel, linewidth=3)
    ax1.plot(ax, ay, 'r-', label=alabel, linewidth=3)
    ax1.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(color='black', linestyle=':')
    plt.title(plot_title)
    saver_helper(fig1, file_name=plot_dir + plot_file)
    plt_free()


def plot_cost_history(plot_dir, datafile, plot_title, plot_file):
    """
    #
    :param plot_dir:
    :param datafile:
    :param plot_title:
    :param plot_file:
    :return:
    """
    # load data
    original_data = load_csv(datafile=plot_dir + datafile)
    x = column(original_data, 0)
    y = column(original_data, 1)
    xlabel = "iterations"
    ylabel = "Cost J(iterations)"
    # plotting
    print_info(title=plot_file, location=plot_dir)
    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, 'g-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(color='black', linestyle=':')
    plt.title(plot_title)
    saver_helper(fig1, file_name=plot_dir + plot_file)
    plt_free()


def qqplot(plot_dir, datafile, plot_title, plot_file):
    """
    #
    :param plot_dir:
    :param datafile:
    :param plot_title:
    :param plot_file:
    :return:
    """
    # load data
    original_data = load_csv(datafile=plot_dir + datafile)
    fitting_data = load_csv(datafile=plot_dir + datafile)
    lx = column(original_data, 0)
    ly = column(original_data, 1)
    ax = column(fitting_data, 2)
    ay = column(fitting_data, 3)
    llabel = "QQplot"
    alabel = "linear"
    xlabel = "estimated"
    ylabel = "samples"
    # plotting
    print_info(title=datafile, location=plot_dir)
    fig1, ax1 = plt.subplots()
    ax1.plot(lx, ly, 'o', color="darkblue", label=llabel, linewidth=3)
    ax1.plot(ax, ay, 'r-', label=alabel, linewidth=2)
    ax1.legend(loc='best')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(color='black', linestyle=':')
    plt.title(plot_title)
    saver_helper(fig1, file_name=plot_dir + plot_file)
    plt_free()


def plot_correlation(plot_dir, datafile, title, plotfile):
    """
    #
    :param plot_dir:
    :param datafile:
    :param title:
    :param plotfile:
    :return:
    """
    # load hurst data
    hustdata = load_csv_str(datafile=plot_dir + datafile)
    bars1 = [float(number) for number in column(hustdata, 1)]
    yer1 = [float(number) for number in column(hustdata, 2)]
    xticks = column(hustdata, 0)
    hline = 1
    # width of the bars
    bar_width = 0.3
    ylabel = 'Correlation'
    # plotting
    print_info(title=datafile, location=plot_dir)
    xpos = np.arange(len(bars1))
    fig, ax = plt.subplots()
    ax.bar(xpos, bars1, width=bar_width, color='yellow', edgecolor='black', yerr=yer1, capsize=7)
    plt.axhline(hline, color="red")
    plt.xticks(xpos, xticks, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    saver_helper(fig, file_name=plot_dir + plotfile)
    plt_free()


def plot_hurst(plot_dir, datafile, title, plotfile):
    """
    #
    :param plot_dir:
    :param datafile:
    :param title:
    :param plotfile:
    :return:
    """
    # load hurst data
    hustdata = load_csv_str(datafile=plot_dir + datafile)
    bars1 = [float(number) for number in column(hustdata, 1)]
    yer1 = [float(number) for number in column(hustdata, 2)]
    xticks = column(hustdata, 0)
    # filter data
    hline = bars1[0]
    bars1 = bars1[1:]
    yer1 = yer1[1:]
    xticks = xticks[1:]
    # width of the bars
    bar_width = 0.3
    ylabel = 'Hurst Exponent'
    print_info(title=datafile, location=plot_dir)
    xpos = np.arange(len(bars1))
    fig, ax = plt.subplots()
    ax.bar(xpos, bars1, width=bar_width, color='cyan', edgecolor='black', yerr=yer1, capsize=7)
    ax.text(1.02, hline, str(hline), va='center', ha="left", bbox=dict(facecolor="w", alpha=0.5),
            transform=ax.get_yaxis_transform())
    plt.axhline(hline, color="red")
    plt.xticks(xpos, xticks, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    saver_helper(fig, file_name=plot_dir + plotfile)
    plt_free()


def plot_std_dev(plot_dir, datafile, title, plotfile):
    """
    #
    :param plot_dir:
    :param datafile:
    :param title:
    :param plotfile:
    :return:
    """
    # load hurst data
    hustdata = load_csv_str(datafile=plot_dir + datafile)
    bars1 = [float(number) for number in column(hustdata, 1)]
    yer1 = [float(number) for number in column(hustdata, 2)]
    xticks = column(hustdata, 0)
    # filter data
    hline = bars1[0]
    bars1 = bars1[1:]
    yer1 = yer1[1:]
    xticks = xticks[1:]
    # width of the bars
    bar_width = 0.3
    ylabel = 'Standard Deviation'
    print_info(title=datafile, location=plot_dir)
    xpos = np.arange(len(bars1))
    fig, ax = plt.subplots()
    ax.bar(xpos, bars1, width=bar_width, color='lime', edgecolor='black', yerr=yer1, capsize=7)
    ax.text(1.02, hline, str(hline), va='center', ha="left", bbox=dict(facecolor="w", alpha=0.5),
            transform=ax.get_yaxis_transform())
    plt.axhline(hline, color="red")
    plt.xticks(xpos, xticks, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    saver_helper(fig, file_name=plot_dir + plotfile)
    plt_free()


def plot_mean(plot_dir, datafile, title, plotfile):
    """
    #
    :param plot_dir:
    :param datafile:
    :param title:
    :param plotfile:
    :return:
    """
    # load hurst data
    hustdata = load_csv_str(datafile=plot_dir + datafile)
    bars1 = [float(number) for number in column(hustdata, 1)]
    yer1 = [float(number) for number in column(hustdata, 2)]
    xticks = column(hustdata, 0)
    # filter data
    hline = bars1[0]
    bars1 = bars1[1:]
    yer1 = yer1[1:]
    xticks = xticks[1:]
    # width of the bars
    bar_width = 0.3
    ylabel = 'Avarage inter-packet time'
    print_info(title=datafile, location=plot_dir)
    xpos = np.arange(len(bars1))
    fig, ax = plt.subplots()
    ax.bar(xpos, bars1, width=bar_width, color='magenta', edgecolor='black', yerr=yer1, capsize=7)
    ax.text(1.02, hline, str(hline), va='center', ha="left", bbox=dict(facecolor="w", alpha=0.5),
            transform=ax.get_yaxis_transform())
    plt.axhline(hline, color="red")
    plt.xticks(xpos, xticks, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    saver_helper(fig, file_name=plot_dir + plotfile)
    plt_free()


def plot_cost_function(plot_dir, datafile, title, plotfile):
    """
    #
    :param plot_dir:
    :param datafile:
    :param title:
    :param plotfile:
    :return:
    """
    # load hurst data
    hustdata = load_csv_str(datafile=plot_dir + datafile)
    bars1 = [float(number) for number in column(hustdata, 1)]
    xticks = column(hustdata, 0)
    bar_width = 0.3
    ylabel = 'Correlation'
    print_info(title=datafile, location=plot_dir)
    # The x position of bars
    xpos = np.arange(len(bars1))
    fig, ax = plt.subplots()
    # Create
    ax.bar(xpos, bars1, width=bar_width, color='purple', edgecolor='black', capsize=7)
    plt.xticks(xpos, xticks, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    saver_helper(fig, file_name=plot_dir + plotfile)
    plt_free()


def _get_ModelValueByFunction(functionIndex, costfunction_data1, costfunction_data2, costfunction_data3,
                              costfunction_data4):
    """

    :param functionIndex:
    :param costfunction_data1:
    :param costfunction_data2:
    :param costfunction_data3:
    :param costfunction_data4:
    :return:
    """
    value1 = float(costfunction_data1[functionIndex][1])
    value2 = float(costfunction_data2[functionIndex][1])
    value3 = float(costfunction_data3[functionIndex][1])
    value4 = float(costfunction_data4[functionIndex][1])
    label = costfunction_data1[functionIndex][0]
    values = [value1, value2, value3, value4]
    ModelCostByFunction = collections.namedtuple('ModelCostByFunction', ['label', 'values'])
    m = ModelCostByFunction(label=label, values=values)
    return m


def plot_cost_function_all2(costfunction1="", costfunction2="", costfunction3="", costfunction4="",
                            pcapname1="", pcapname2="", pcapname3="", pcapname4="",
                            title="title", plotfile="plotfile"):
    """

    :param costfunction1:
    :param costfunction2:
    :param costfunction3:
    :param costfunction4:
    :param pcapname1:
    :param pcapname2:
    :param pcapname3:
    :param pcapname4:
    :param title:
    :param plotfile:
    :return:
    """
    print_info(title=title, location=plotfile)
    line_width = 2.2
    mark_size = 8
    costfunction_data1 = order_matrix_str(load_csv_str(datafile=costfunction1), 0)
    costfunction_data2 = order_matrix_str(load_csv_str(datafile=costfunction2), 0)
    costfunction_data3 = order_matrix_str(load_csv_str(datafile=costfunction3), 0)
    costfunction_data4 = order_matrix_str(load_csv_str(datafile=costfunction4), 0)
    m0 = _get_ModelValueByFunction(0, costfunction_data1, costfunction_data2, costfunction_data3, costfunction_data4)
    m1 = _get_ModelValueByFunction(1, costfunction_data1, costfunction_data2, costfunction_data3, costfunction_data4)
    m2 = _get_ModelValueByFunction(2, costfunction_data1, costfunction_data2, costfunction_data3, costfunction_data4)
    m3 = _get_ModelValueByFunction(3, costfunction_data1, costfunction_data2, costfunction_data3, costfunction_data4)
    m4 = _get_ModelValueByFunction(4, costfunction_data1, costfunction_data2, costfunction_data3, costfunction_data4)
    m5 = _get_ModelValueByFunction(5, costfunction_data1, costfunction_data2, costfunction_data3, costfunction_data4)
    m6 = _get_ModelValueByFunction(6, costfunction_data1, costfunction_data2, costfunction_data3, costfunction_data4)
    label0 = m0.label
    label1 = m1.label
    label2 = m2.label
    label3 = m3.label
    label4 = m4.label
    label5 = m5.label
    label6 = m6.label
    values0 = m0.values
    values1 = m1.values
    values2 = m2.values
    values3 = m3.values
    values4 = m4.values
    values5 = m5.values
    values6 = m6.values
    xlables = [pcapname1, pcapname2, pcapname3, pcapname4]
    xvalues = [1, 2, 3, 4]
    fig, ax = plt.subplots()
    with plt.style.context('default'):
        plt.xticks(xvalues, xlables)
        plt.plot(xvalues, values0, label=label0, marker='o', linewidth=line_width, markersize=mark_size, color="darkblue", markeredgecolor="black")
        plt.plot(xvalues, values1, label=label1, marker='*', linewidth=line_width, markersize=mark_size, color="dodgerblue", markeredgecolor="black")
        plt.plot(xvalues, values2, label=label2, marker='d', linewidth=line_width, markersize=mark_size, color="springgreen", markeredgecolor="black")
        plt.plot(xvalues, values3, label=label3, marker='<', linewidth=line_width, markersize=mark_size, color="lime", markeredgecolor="black")
        plt.plot(xvalues, values4, label=label4, marker='>', linewidth=line_width, markersize=mark_size, color="gold", markeredgecolor="black")
        plt.plot(xvalues, values5, label=label5, marker='s', linewidth=line_width, markersize=mark_size, color="red", markeredgecolor="black")
        plt.plot(xvalues, values6, label=label6, marker='P', linewidth=line_width, markersize=mark_size, color="purple", markeredgecolor="black")
    # Shrink current axis's height by 10% on the bottom
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=True, shadow=True, ncol=3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_aspect(aspect=0.10)
    plt.plot(usetex=True)
    plt.ylabel("Cost Function $J$")
    plt.xlabel('Pcap Files')
    plt.title(title)
    # plt.legend()
    plt.grid(color='black', linestyle=':', axis='y')
    plt.tight_layout()
    # plt.legend()
    # plt.show()
    saver_helper(fig, file_name=plotfile)
    plt_free()


def plot_aic_bic2(aicbicfile1="", aicbicfile2="", aicbicfile3="", aicbicfile4="",
                  pcapname1="", pcapname2="", pcapname3="", pcapname4="",
                  title="title", plotfile="plotfile"):
    """

    :param aicbicfile1:
    :param aicbicfile2:
    :param aicbicfile3:
    :param aicbicfile4:
    :param pcapname1:
    :param pcapname2:
    :param pcapname3:
    :param pcapname4:
    :param title:
    :param plotfile:
    :return:
    """
    print_info(title=title, location=plotfile)
    line_width = 2
    mark_size = 8
    # ---- Plot the order for each pcap
    aicbic1 = load_csv_str(datafile=aicbicfile1)
    aicbic2 = load_csv_str(datafile=aicbicfile2)
    aicbic3 = load_csv_str(datafile=aicbicfile3)
    aicbic4 = load_csv_str(datafile=aicbicfile4)
    aicbic1 = column(order_matrix(aicbic1, 1), 0)
    aicbic2 = column(order_matrix(aicbic2, 1), 0)
    aicbic3 = column(order_matrix(aicbic3, 1), 0)
    aicbic4 = column(order_matrix(aicbic4, 1), 0)
    aicbicorder1 = order_matrix_str([[aicbic1[i], i + 1] for i in range(0, len(aicbic1))], 0)
    aicbicorder2 = order_matrix_str([[aicbic2[i], i + 1] for i in range(0, len(aicbic2))], 0)
    aicbicorder3 = order_matrix_str([[aicbic3[i], i + 1] for i in range(0, len(aicbic3))], 0)
    aicbicorder4 = order_matrix_str([[aicbic4[i], i + 1] for i in range(0, len(aicbic4))], 0)
    m0 = _get_ModelValueByFunction(0, aicbicorder1, aicbicorder2, aicbicorder3, aicbicorder4)
    m1 = _get_ModelValueByFunction(1, aicbicorder1, aicbicorder2, aicbicorder3, aicbicorder4)
    m2 = _get_ModelValueByFunction(2, aicbicorder1, aicbicorder2, aicbicorder3, aicbicorder4)
    m3 = _get_ModelValueByFunction(3, aicbicorder1, aicbicorder2, aicbicorder3, aicbicorder4)
    m4 = _get_ModelValueByFunction(4, aicbicorder1, aicbicorder2, aicbicorder3, aicbicorder4)
    m5 = _get_ModelValueByFunction(5, aicbicorder1, aicbicorder2, aicbicorder3, aicbicorder4)
    m6 = _get_ModelValueByFunction(6, aicbicorder1, aicbicorder2, aicbicorder3, aicbicorder4)
    label0 = m0.label
    label1 = m1.label
    label2 = m2.label
    label3 = m3.label
    label4 = m4.label
    label5 = m5.label
    label6 = m6.label
    values0 = m0.values
    values1 = m1.values
    values2 = m2.values
    values3 = m3.values
    values4 = m4.values
    values5 = m5.values
    values6 = m6.values
    xlables = [str(pcapname1), str(pcapname2), str(pcapname3), str(pcapname4)]
    xvalues = [1, 2, 3, 4]
    fig, ax = plt.subplots()
    with plt.style.context('default'):
        plt.xticks(xvalues, xlables)
        plt.plot(xvalues, values0, label=label0, marker='o', linewidth=line_width, markersize=mark_size, color="darkblue", markeredgecolor="black")
        plt.plot(xvalues, values1, label=label1, marker='*', linewidth=line_width, markersize=mark_size, color="dodgerblue", markeredgecolor="black")
        plt.plot(xvalues, values2, label=label2, marker='d', linewidth=line_width, markersize=mark_size, color="springgreen", markeredgecolor="black")
        plt.plot(xvalues, values3, label=label3, marker='<', linewidth=line_width, markersize=mark_size, color="lime", markeredgecolor="black")
        plt.plot(xvalues, values4, label=label4, marker='>', linewidth=line_width, markersize=mark_size, color="gold", markeredgecolor="black")
        plt.plot(xvalues, values5, label=label5, marker='s', linewidth=line_width, markersize=mark_size, color="red", markeredgecolor="black")
        plt.plot(xvalues, values6, label=label6, marker='P', linewidth=line_width, markersize=mark_size, color="purple", markeredgecolor="black")
    # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    # Put a legend below current axis
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), fancybox=True, shadow=True, ncol=3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_aspect(aspect=0.25)
    plt.ylabel("AIC/BIC position")
    plt.xlabel('Pcap Files')
    plt.title(title)
    plt.grid(color='black', linestyle=':', axis='y')
    plt.tight_layout()
    # plt.show()
    saver_helper(fig, file_name=plotfile)
    plt_free()


""""
    aicbic_data1 = order_matrix_str(load_csv_str(datafile=aicbicfile1), 0)
    aicbic_data2 = order_matrix_str(load_csv_str(datafile=aicbicfile2), 0)
    aicbic_data3 = order_matrix_str(load_csv_str(datafile=aicbicfile3), 0)
    aicbic_data4 = order_matrix_str(load_csv_str(datafile=aicbicfile4), 0)
    print('costfunction_data1: ' + str(aicbic_data1))
    print('costfunction_data2: ' + str(aicbic_data2))
    print('costfunction_data3: ' + str(aicbic_data3))
    print('costfunction_data4: ' + str(aicbic_data4)
"""
"""
    m = _get_ModelCostByFunction(0, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    print("m.label:"+m.label)
    print("m.values:" + str(m.values))
    m0 = _get_ModelCostByFunction(0, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    m1 = _get_ModelCostByFunction(1, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    m2 = _get_ModelCostByFunction(2, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    m3 = _get_ModelCostByFunction(3, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    m4 = _get_ModelCostByFunction(4, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    m5 = _get_ModelCostByFunction(5, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    m6 = _get_ModelCostByFunction(6, aicbic_data1, aicbic_data2, aicbic_data3, aicbic_data4)
    label0 = m0.label
    label1 = m1.label
    label2 = m2.label
    label3 = m3.label
    label4 = m4.label
    label5 = m5.label
    label6 = m6.label
    values0 = m0.values
    values1 = m1.values
    values2 = m2.values
    values3 = m3.values
    values4 = m4.values
    values5 = m5.values
    values6 = m6.values
    xlables = [pcapname1, pcapname2, pcapname3, pcapname4]
    xvalues = [1, 2, 3, 4]
    fig, ax = plt.subplots()
    with plt.style.context('default'):
        plt.xticks(xvalues, xlables)
        plt.plot(xvalues, values0, label=label0, marker='o')
        plt.plot(xvalues, values1, label=label1, marker='*')
        plt.plot(xvalues, values2, label=label2, marker='d')
        plt.plot(xvalues, values3, label=label3, marker='<')
        plt.plot(xvalues, values4, label=label4, marker='>')
        plt.plot(xvalues, values5, label=label5, marker='s')
        plt.plot(xvalues, values6, label=label6, marker='P')
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3)
    plt.ylabel("AIC/BIC position")
    plt.xlabel('Model')
    plt.title(title)
    #plt.legend()
    plt.grid(color='black', linestyle=':', axis='y')
    plt.tight_layout()
    #plt.legend()
    plt.show()
    saver_helper(fig, file_name='aic-bic-pos')
    plt_free()
"""


def plot_cost_function_all(costfunction1="", costfunction2="", costfunction3="", costfunction4="",
                           pcapname1="", pcapname2="", pcapname3="", pcapname4="",
                           title="title", plotfile="plotfile"):
    """
    #
    :param costfunction1:
    :param costfunction2:
    :param costfunction3:
    :param costfunction4:
    :param pcapname1:
    :param pcapname2:
    :param pcapname3:
    :param pcapname4:
    :param title:
    :param plotfile:
    :return:
    """
    print_info(title=title, location=plotfile)
    costfunction_data1 = order_matrix_str(load_csv_str(datafile=costfunction1), 0)
    costfunction_data2 = order_matrix_str(load_csv_str(datafile=costfunction2), 0)
    costfunction_data3 = order_matrix_str(load_csv_str(datafile=costfunction3), 0)
    costfunction_data4 = order_matrix_str(load_csv_str(datafile=costfunction4), 0)
    bars1 = [float(i) for i in column(costfunction_data1, 1)]
    bars2 = [float(i) for i in column(costfunction_data2, 1)]
    bars3 = [float(i) for i in column(costfunction_data3, 1)]
    bars4 = [float(i) for i in column(costfunction_data4, 1)]
    xticks = column(costfunction_data1, 0)
    bar_width = 0.2
    ylabel = 'Cost Function'
    bar1label = pcapname1
    bar2label = pcapname2
    bar3label = pcapname3
    bar4label = pcapname4
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2 * bar_width for x in r1]
    r4 = [x + 3 * bar_width for x in r1]
    fig, ax = plt.subplots()
    ax.bar(r1, bars1, width=bar_width, color='springgreen', hatch="////", lw=1, edgecolor='#003300', capsize=7,
           label=bar1label)
    ax.bar(r2, bars2, width=bar_width, color='fuchsia', hatch="\\\\\\\\", lw=1, edgecolor='#660066', capsize=7,
           label=bar2label)
    ax.bar(r3, bars3, width=bar_width, color='mediumblue', hatch="----", lw=1, edgecolor='#000066', capsize=7,
           label=bar3label)
    ax.bar(r4, bars4, width=bar_width, color='gold', hatch="xxxx", lw=1, edgecolor='#666600', capsize=7,
           label=bar4label)
    plt.xticks([r + bar_width for r in range(len(bars1))], xticks, rotation=45)
    plt.ylabel(ylabel)
    plt.xlabel('model')
    plt.title(title)
    plt.legend()
    plt.grid(color='black', linestyle=':', axis='y')
    plt.tight_layout()
    saver_helper(fig, file_name=plotfile)
    plt_free()


"""
def plot_cost_function_all_pcap():
    PLOT_DIR = "./plots/"
    costfunction1 = PLOT_DIR + "skype/costFunction.dat"
    pcaptitle1 = "skype"
    costfunction2 = PLOT_DIR + "bigFlows/costFunction.dat"
    pcaptitle2 = "lan gateway"
    costfunction3 = PLOT_DIR + "lanDiurnal2/costFunction.dat"
    pcaptitle3 = "lan firewall diurnal"
    costfunction4 = PLOT_DIR + "equinix-1s/costFunction.dat"
    pcaptitle4 = "wan"
    costfunction_data1 = order_matrix(load_csv_str(datafile=costfunction1), 1)
    costfunction_data2 = order_matrix(load_csv_str(datafile=costfunction2), 1)
    costfunction_data3 = order_matrix(load_csv_str(datafile=costfunction3), 1)
    costfunction_data4 = order_matrix(load_csv_str(datafile=costfunction4), 1)
    data1 = create_plot_data_costfunction(costfunction_data1)
    data2 = create_plot_data_costfunction(costfunction_data2)
    data3 = create_plot_data_costfunction(costfunction_data3)
    data4 = create_plot_data_costfunction(costfunction_data4)
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax00 = axs[0][0].bar(column(data1, 0), column(data1, 1))
    ax10 = axs[1][0].bar(column(data2, 0), column(data2, 1))
    ax01 = axs[0][1].bar(column(data3, 0), column(data3, 1))
    ax11 = axs[1][1].bar(column(data4, 0), column(data4, 1))
    ax11.plt.tight_layout(45)
    ax11.set_horizontalalignment("right")
    #plt.rotation = 45
    for i in range(0, len(ax00)):
        ax00[i].set_color(column(data1, 2)[i])
        ax10[i].set_color(column(data2, 2)[i])
        ax01[i].set_color(column(data3, 2)[i])
        ax11[i].set_color(column(data4, 2)[i])
    # for label in axs[0][0].get_xmajorticklabels() + axs[0][1].get_xmajorticklabels() + \
    #         axs[1][0].get_xmajorticklabels() + axs[1][1].get_xmajorticklabels():
    #    label.set_rotation(45)
    #   label.set_horizontalalignment("right")
    #ax00[0].set_color('r')
    # ax.bar([1,2,3,4], [1,2,3,4])
    plt.tight_layout()
    plt.show()
    print(costfunction_data1)
    print(costfunction_data2)
    print(costfunction_data3)
    print(costfunction_data4)
"""

"""
def map_model_color(model):
    color = "white"
    if model == "Cauchy":
        color = "red"
    elif model == "Exponential(LR)":
        color = "dodgerblue"
    elif model == "Exponential(Me)":
        color = "yellow"
    elif model == "Normal":
        color = "blue"
    elif model == "Pareto(LR)":
        color = "limegreen"
    elif model == "Pareto(MLH)":
        color = "deeppink"
    elif model == "Weibull":
        color = "indigo"
    else:
        print("Error, model not found: "+str(model))
    return color
"""

"""
def create_plot_data_costfunction(costfunction_vector):
    data_vector = []
    for i in range (0, len(costfunction_vector)):
        data1 = []
        data1.append(costfunction_vector[i][0])
        data1.append(float(costfunction_vector[i][1]))
        data1.append(map_model_color(costfunction_vector[i][0]))
        data_vector.append(data1)
    return data_vector
"""


def plot_costfunction_vs_aicbic(aicbic1, costfunction1, pcaptitle1,
                                aicbic2, costfunction2, pcaptitle2,
                                aicbic3, costfunction3, pcaptitle3,
                                aicbic4, costfunction4, pcaptitle4,
                                title, plotfile):
    """
    Plots the relative difference between the model accuracy order of the cost
    function and AIC/BIC order of model selection
    :param aicbic1:
    :param costfunction1:
    :param pcaptitle1:
    :param aicbic2:
    :param costfunction2:
    :param pcaptitle2:
    :param aicbic3:
    :param costfunction3:
    :param pcaptitle3:
    :param aicbic4:
    :param costfunction4:
    :param pcaptitle4:
    :param title:
    :param plotfile:
    :return:
    """
    print_info(title=title, location=plotfile)
    # create AIC and BIC csv file
    # table2csv(aicbic1)
    # table2csv(aicbic2)
    # table2csv(aicbic3)
    # table2csv(aicbic4)
    # aicbic_data1 = load_csv_str(datafile=aicbic1 + '.csv')
    # aicbic_data2 = load_csv_str(datafile=aicbic2 + '.csv')
    # aicbic_data3 = load_csv_str(datafile=aicbic3 + '.csv')
    # aicbic_data4 = load_csv_str(datafile=aicbic4 + '.csv')
    aicbic_data1 = load_csv_str(datafile=aicbic1)
    aicbic_data2 = load_csv_str(datafile=aicbic2)
    aicbic_data3 = load_csv_str(datafile=aicbic3)
    aicbic_data4 = load_csv_str(datafile=aicbic4)
    costfunction_data1 = load_csv_str(datafile=costfunction1)
    costfunction_data2 = load_csv_str(datafile=costfunction2)
    costfunction_data3 = load_csv_str(datafile=costfunction3)
    costfunction_data4 = load_csv_str(datafile=costfunction4)
    # plot
    # print("* " + pcaptitle1)
    bars1 = calc_relative_position_rank_diff(costfunction_data1, aicbic_data1)
    # print("* " + pcaptitle2)
    bars2 = calc_relative_position_rank_diff(costfunction_data2, aicbic_data2)
    # print("* " + pcaptitle3)
    bars3 = calc_relative_position_rank_diff(costfunction_data3, aicbic_data3)
    # print("* " + pcaptitle4)
    bars4 = calc_relative_position_rank_diff(costfunction_data4, aicbic_data4)
    # Horizontal line
    hline1 = (len(costfunction_data4) - 1) / 2
    hline2 = -hline1
    xticks = ['0', '1', '2', '3', '4', '5', '6']
    bar_width = 0.2
    bar1label = pcaptitle1
    bar2label = pcaptitle2
    bar3label = pcaptitle3
    bar4label = pcaptitle4
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2 * bar_width for x in r1]
    r4 = [x + 3 * bar_width for x in r1]
    fig, ax = plt.subplots()
    # Create blue bars #ff0000
    ax.bar(r1, bars1, width=bar_width, color='springgreen', hatch="////", lw=1, edgecolor='#003300', capsize=7,
           label=bar1label)
    ax.bar(r2, bars2, width=bar_width, color='fuchsia', hatch="\\\\\\\\", lw=1, edgecolor='#660066', capsize=7,
           label=bar2label)
    ax.bar(r3, bars3, width=bar_width, color='mediumblue', hatch="----", lw=1, edgecolor='#000066', capsize=7,
           label=bar3label)
    ax.bar(r4, bars4, width=bar_width, color='gold', hatch="xxxx", lw=1, edgecolor='#666600', capsize=7,
           label=bar4label)
    # create hline
    plt.axhline(hline1, color="red")
    plt.axhline(hline2, color="red")
    # general layout
    plt.xticks([r + bar_width for r in range(len(bars1))], xticks)
    plt.ylabel('Ranking delta $\delta$')
    plt.xlabel('n-th best model according to $J$')
    plt.title(title)
    plt.legend()
    plt.grid(color='black', linestyle=':', axis='y')
    saver_helper(fig, file_name=plotfile)
    plt_free()


def plot_aic_bic(aicbicfile1, pcaptitle1, aicbicfile2, pcaptitle2,
                 aicbicfile3, pcaptitle3, aicbicfile4, pcaptitle4,
                 title_sumary, plotfile_sumary, title_order, plotfile_order):
    """
    :param aicbicfile1:
    :param pcaptitle1:
    :param aicbicfile2:
    :param pcaptitle2:
    :param aicbicfile3:
    :param pcaptitle3:
    :param aicbicfile4:
    :param pcaptitle4:
    :param title_sumary:
    :param plotfile_sumary:
    :param title_order:
    :param plotfile_order:
    :return:
    """
    aicbic1 = load_csv_str(datafile=aicbicfile1)
    aicbic2 = load_csv_str(datafile=aicbicfile2)
    aicbic3 = load_csv_str(datafile=aicbicfile3)
    aicbic4 = load_csv_str(datafile=aicbicfile4)
    # ---- Plot in log scale
    bars1 = signedlog([float(i) for i in column(aicbic1, 1)])
    bars2 = signedlog([float(i) for i in column(aicbic2, 1)])
    bars3 = signedlog([float(i) for i in column(aicbic3, 1)])
    bars4 = signedlog([float(i) for i in column(aicbic4, 1)])
    xticks = column(aicbic1, 0)
    bar_width = 0.2
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2 * bar_width for x in r1]
    r4 = [x + 3 * bar_width for x in r1]
    fig, ax = plt.subplots()
    ax.bar(r1, bars1, width=bar_width, color='springgreen', hatch="////", lw=1, edgecolor='#003300', capsize=7,
           label=pcaptitle1)
    ax.bar(r2, bars2, width=bar_width, color='fuchsia', hatch="\\\\\\\\", lw=1, edgecolor='#660066', capsize=7,
           label=pcaptitle2)
    ax.bar(r3, bars3, width=bar_width, color='mediumblue', hatch="----", lw=1, edgecolor='#000066', capsize=7,
           label=pcaptitle3)
    ax.bar(r4, bars4, width=bar_width, color='gold', hatch="xxxx", lw=1, edgecolor='#666600', capsize=7,
           label=pcaptitle4)
    plt.xticks([r + bar_width for r in range(len(bars1))], xticks, rotation=45)
    plt.ylabel('(Aic/|Aic|)*ln|Aic|')
    plt.xlabel('model')
    plt.title(title_sumary)
    plt.legend()
    plt.tight_layout()
    plt.grid(color='black', linestyle=':', axis='y')
    print_info(title=title_sumary, location=plotfile_sumary)
    saver_helper(fig, file_name=plotfile_sumary)
    plt_free()
    # ---- Plot the order for each pcap
    aicbic1 = column(order_matrix(aicbic1, 1), 0)
    aicbic2 = column(order_matrix(aicbic2, 1), 0)
    aicbic3 = column(order_matrix(aicbic3, 1), 0)
    aicbic4 = column(order_matrix(aicbic4, 1), 0)
    aicbicorder1 = order_matrix_str([[aicbic1[i], i + 1] for i in range(0, len(aicbic1))], 0)
    aicbicorder2 = order_matrix_str([[aicbic2[i], i + 1] for i in range(0, len(aicbic2))], 0)
    aicbicorder3 = order_matrix_str([[aicbic3[i], i + 1] for i in range(0, len(aicbic3))], 0)
    aicbicorder4 = order_matrix_str([[aicbic4[i], i + 1] for i in range(0, len(aicbic4))], 0)
    bars1 = [float(i) for i in column(aicbicorder1, 1)]
    bars2 = [float(i) for i in column(aicbicorder2, 1)]
    bars3 = [float(i) for i in column(aicbicorder3, 1)]
    bars4 = [float(i) for i in column(aicbicorder4, 1)]
    xticks = column(aicbicorder1, 0)
    bar_width = 0.2
    # The x position of bars
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2 * bar_width for x in r1]
    r4 = [x + 3 * bar_width for x in r1]
    fig, ax = plt.subplots()
    ax.bar(r1, bars1, width=bar_width, color='springgreen', hatch="////", lw=1, edgecolor='#003300', capsize=7,
           label=pcaptitle1)
    ax.bar(r2, bars2, width=bar_width, color='fuchsia', hatch="\\\\\\\\", lw=1, edgecolor='#660066', capsize=7,
           label=pcaptitle2)
    ax.bar(r3, bars3, width=bar_width, color='mediumblue', hatch="----", lw=1, edgecolor='#000066', capsize=7,
           label=pcaptitle3)
    ax.bar(r4, bars4, width=bar_width, color='gold', hatch="xxxx", lw=1, edgecolor='#666600', capsize=7,
           label=pcaptitle4)
    plt.xticks([r + bar_width for r in range(len(bars1))], xticks, rotation=45)
    plt.ylabel('AIC and BIC ranking')
    plt.title(title_order)
    plt.legend()
    plt.grid(color='black', linestyle=':', axis='y')
    plt.tight_layout()
    print_info(title=title_order, location=plotfile_order)
    saver_helper(fig, file_name=plotfile_order)
    plt_free()


def plot_lanfirewall_cdfs(plots_root, data_dir, plot_dir, empirical_cdf_crossval, pareto_lr_cdf_approximation,
                               pareto_mlh_cdf_approximation, weibull_cdf_approximation):

    plot_title = "Approximations vs Cross-validation dataset"
    datadir = plots_root + data_dir
    plotdir = plots_root + plot_dir
    print_info(title=plot_title + "(from lan-firewall pcap)", location=plotdir)
    # datafiles
    empirical_cdf_crossval = datadir + empirical_cdf_crossval
    pareto_lr_cdf_approximation = datadir + pareto_lr_cdf_approximation
    pareto_mlh_cdf_approximation = datadir + pareto_mlh_cdf_approximation
    weibull_cdf_approximation = datadir + weibull_cdf_approximation
    # load data
    original_data = load_csv(empirical_cdf_crossval)
    pa_lr = load_csv(pareto_lr_cdf_approximation)
    pa_mlh = load_csv(pareto_mlh_cdf_approximation)
    weibull = load_csv(weibull_cdf_approximation)
    ox = column(original_data, 0)
    oy = column(original_data, 1)
    px = column(pa_mlh, 0)
    py = column(pa_mlh, 1)
    plx = column(pa_lr, 0)
    ply = column(pa_lr, 1)
    wx = column(weibull, 0)
    wy = column(weibull, 1)
    # Plot Figure
    fig2, ax2 = plt.subplots()
    xlabel = "Inter packet time (s)"
    ylabel = "CDF function"
    ax2.plot(ox, oy, '-', color="red", label="cross-validation", linewidth=2)
    ax2.plot(plx, ply, ':', color="green", label="Pareto(LR)", linewidth=3)
    ax2.plot(px, py, '-.', color="purple", label="Pareto(MLH)", linewidth=3)
    ax2.plot(wx, wy, '--', color="darkblue", label="Weibull", linewidth=3)
    ax2.legend(loc='upper left')
    plt.semilogx()
    plt.ylim([0, 1.01])
    plt.grid(color='black', linestyle=':')
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    saver_helper(fig2, file_name=plotdir + "cdfs")
    plt_free()


########################################################################################################################
# Plot scripts
########################################################################################################################

def dataprocessor_simulation_plot(plot_dir):
    """
    #
    :param plot_dir:
    :return:
    """
    # Plot CDF fitting
    term.print_h1(title="Plot CDF fitting")
    # Weibull
    plot_cdf_fitting(plot_dir=plot_dir, fitting_data='Weibull aproximation vs Original set.dat',
                     original_datafile='Empirical CDF function.dat',
                     plot_title='Weibull aproximation vs Original set',
                     plot_file='Weibull aproximation vs Original set')
    # Exponential (LR)
    plot_cdf_fitting(plot_dir=plot_dir, fitting_data="Exponential aproximation (linear regression) vs Original set.dat",
                     original_datafile="Empirical CDF function.dat",
                     plot_title="Exponential aproximation (linear regression) vs Original set",
                     plot_file="Exponential aproximation (linear regression) vs Original set")
    # Exponential (Me)
    plot_cdf_fitting(plot_dir=plot_dir, fitting_data="Exponential aproximation (mean) vs Original set.dat",
                     original_datafile="Empirical CDF function.dat",
                     plot_title="Exponential aproximation (mean) vs Original set",
                     plot_file="Exponential aproximation (mean) vs Original set")
    # Normal
    plot_cdf_fitting(plot_dir=plot_dir, fitting_data="Normal aproximation vs Original set.dat",
                     original_datafile="Empirical CDF function.dat",
                     plot_title="Normal aproximation vs Original set",
                     plot_file="Normal aproximation vs Original set")
    # Pareto (LR)
    plot_cdf_fitting(plot_dir=plot_dir, fitting_data="Pareto aproximation (linear regression) vs Original set.dat",
                     original_datafile="Empirical CDF function.dat",
                     plot_title="Pareto aproximation (linear regression) vs Original set",
                     plot_file="Pareto aproximation (linear regression) vs Original set")
    # Pareto (MLH)
    plot_cdf_fitting(plot_dir=plot_dir, fitting_data="Pareto aproximation (maximum likehood) vs Original set.dat",
                     original_datafile="Empirical CDF function.dat",
                     plot_title="Pareto aproximation (maximum likehood) vs Original set",
                     plot_file="Pareto aproximation (maximum likehood) vs Original set")
    # Cauchy
    plot_cdf_fitting(plot_dir=plot_dir, fitting_data="Cauchy aproximation vs Original set.dat",
                     original_datafile="Empirical CDF function.dat",
                     plot_title="Cauchy aproximation vs Original set",
                     plot_file="Cauchy aproximation vs Original set")

    term.print_h1("Plot Linear Regression graphs")
    # Weibull
    plot_linear_regression(plot_dir=plot_dir, datafile="Weibull - Linearized data and linear fitting.dat",
                           plot_title="Weibull - Linearized data and linear fitting",
                           plot_file="Weibull - Linearized data and linear fitting")
    plot_cost_history(plot_dir=plot_dir, datafile="Weibull - Cost J(iterations) convergence.dat",
                      plot_title="Weibull: Linear Regression Cost History",
                      plot_file="Weibull - Cost J(iterations) convergence")
    # Exponential (LR)
    plot_linear_regression(plot_dir=plot_dir, datafile="Exponential - Linearized data and linear fitting.dat",
                           plot_title="Exponential - Linearized data and linear fitting",
                           plot_file="Exponential - Linearized data and linear fitting")
    plot_cost_history(plot_dir=plot_dir, datafile="Exponential - Cost J(iterations) convergence.dat",
                      plot_title="Exponential: Linear Regression Cost History",
                      plot_file="Exponential - Cost J(iterations) convergence")
    # Pareto (LR)
    plot_linear_regression(plot_dir=plot_dir, datafile="Pareto - Linearized data and linear fitting.dat",
                           plot_title="Pareto: Linearized data and linear fitting",
                           plot_file="Pareto - Linearized data and linear fitting")
    plot_cost_history(plot_dir=plot_dir, datafile="Pareto - Cost J(iterations) convergence.dat",
                      plot_title="Pareto: Linear Regression Cost History",
                      plot_file="Pareto - Cost J(iterations) convergence")
    # Cauchy
    plot_linear_regression(plot_dir=plot_dir, datafile="Cauchy - Linearized data and linear fitting.dat",
                           plot_title="Cauchy: Linearized data and linear fitting",
                           plot_file="Cauchy - Linearized data and linear fitting")
    plot_cost_history(plot_dir=plot_dir, datafile="Cauchy - Cost J(iterations) convergence.dat",
                      plot_title="Cauchy: Linearized data and linear fitting",
                      plot_file="Cauchy - Linearized data and linear fitting")
    # Plot QQplots
    term.print_h1("Plot QQplots")
    # Weibull
    qqplot(plot_dir=plot_dir, datafile="QQplot - Weibull.dat", plot_title="QQplot - Weibull",
           plot_file="QQplot - Weibull")
    # Exponential (LR)
    qqplot(plot_dir=plot_dir, datafile="QQplot - Exponential(LR).dat", plot_title="QQplot - Exponential(LR)",
           plot_file="QQplot - Exponential(LR)")
    # Exponential (Me)
    qqplot(plot_dir=plot_dir, datafile="QQplot - Exponential(Me).dat", plot_title="QQplot - Exponential(Me)",
           plot_file="QQplot - Exponential(Me)")
    # Normal
    qqplot(plot_dir=plot_dir, datafile="QQplot - Normal.dat", plot_title="QQplot - Normal",
           plot_file="QQplot - Normal")
    # Pareto (LR)
    qqplot(plot_dir=plot_dir, datafile="QQplot - Pareto(LR).dat", plot_title="QQplot - Pareto(LR)",
           plot_file="QQplot - Pareto(LR)")
    # Pareto (MLH)
    qqplot(plot_dir=plot_dir, datafile="QQplot - Pareto(MLH).dat", plot_title="QQplot - Pareto(MLH)",
           plot_file="QQplot - Pareto(MLH)")
    # Cauchy
    qqplot(plot_dir=plot_dir, datafile="QQplot - Cauchy.dat", plot_title="QQplot - Cauchy",
           plot_file="QQplot - Cauchy")
    # Model Evaluation Plots
    term.print_h1("Model Evaluation Plots")
    plot_hurst(plot_dir, 'Hurst Exponent.dat', 'Hurst Exponent', 'Hurst Exponent')
    plot_mean(plot_dir, 'Mean.dat', 'Avarage inter-packet time', 'Mean')
    plot_std_dev(plot_dir, 'Standard Deviation.dat', 'Standard Deviation', 'Standard Deviation')
    plot_correlation(plot_dir, 'Correlation.dat', 'Correlation', 'Correlation')
    plot_cost_function(plot_dir, 'costFunction.dat', 'Cost Function', 'costFunction')


def paper_aicbic_plots_pt1():
    """
    Generate the plots for AIC/BIC paper
    """
    term.print_h1("AIC/BIC and CostFunction Plots")
    plot_dir = "./plots/"
    term.command(cmd="mkdir -p " + plot_dir + 'paper/', color="green")
    pcaptitle1 = "skype"
    pcaptitle2 = "lan-gateway"
    pcaptitle3 = "lan-firewall"
    pcaptitle4 = "wan"
    costfunction1 = plot_dir + "skype/costFunction.dat"
    costfunction2 = plot_dir + "bigFlows/costFunction.dat"
    costfunction3 = plot_dir + "lan-firewall/costFunction.dat"
    costfunction4 = plot_dir + "equinix-1s/costFunction.dat"
    # AIC and BIC tables
    aicbic1 = plot_dir + "skype/Aic-Bic.dat"
    aicbic2 = plot_dir + "bigFlows/Aic-Bic.dat"
    aicbic3 = plot_dir + "lan-firewall/Aic-Bic.dat"
    aicbic4 = plot_dir + "equinix-1s/Aic-Bic.dat"
    prepare_aic_bic_csv([aicbic1, aicbic2, aicbic3, aicbic4])
    # AIC and BIC csv
    aicbic1 = plot_dir + "skype/Aic-Bic.dat.csv"
    aicbic2 = plot_dir + "bigFlows/Aic-Bic.dat.csv"
    aicbic3 = plot_dir + "lan-firewall/Aic-Bic.dat.csv"
    aicbic4 = plot_dir + "equinix-1s/Aic-Bic.dat.csv"
    # Cost Function Sumary
    plot_cost_function_all(costfunction1=costfunction1, costfunction2=costfunction2,
                           costfunction3=costfunction3, costfunction4=costfunction4,
                           pcapname1=pcaptitle1, pcapname2=pcaptitle2, pcapname3=pcaptitle3,
                           pcapname4=pcaptitle4, title="Cost Function Sumary",
                           plotfile=plot_dir + "paper/cost-function-summary")
    # Cost Function and AIC/BIC relative difference
    plot_costfunction_vs_aicbic(aicbic1=aicbic1, costfunction1=costfunction1,
                                pcaptitle1=pcaptitle1,
                                aicbic2=aicbic2, costfunction2=costfunction2,
                                pcaptitle2=pcaptitle2,
                                aicbic3=aicbic3, costfunction3=costfunction3,
                                pcaptitle3=pcaptitle3,
                                aicbic4=aicbic4, costfunction4=costfunction4,
                                pcaptitle4=pcaptitle4,
                                title="Cost Function and AIC/BIC relative difference",
                                plotfile=plot_dir + "paper/aicbic-costfunction-relative-diff")
    plot_aic_bic(aicbicfile1=aicbic1, pcaptitle1=pcaptitle1,
                 aicbicfile2=aicbic2, pcaptitle2=pcaptitle2,
                 aicbicfile3=aicbic3, pcaptitle3=pcaptitle3,
                 aicbicfile4=aicbic4, pcaptitle4=pcaptitle4,
                 title_sumary="AIC and BIC values", plotfile_sumary=plot_dir + 'paper/aic-bic-logscale-sumary',
                 title_order="AIC and BIC position", plotfile_order=plot_dir + 'paper/aic-bic-order')
    plot_cost_function_all2(costfunction1=costfunction1, costfunction2=costfunction2,
                            costfunction3=costfunction3, costfunction4=costfunction4,
                            pcapname1=pcaptitle1, pcapname2=pcaptitle2, pcapname3=pcaptitle3,
                            pcapname4=pcaptitle4, title="Cost Function Sumary",
                            plotfile=plot_dir + "paper/cost-function-summary-v2")
    plot_aic_bic2(aicbicfile1=aicbic1, aicbicfile2=aicbic2,
                  aicbicfile3=aicbic3, aicbicfile4=aicbic4,
                  pcapname1=pcaptitle1, pcapname2=pcaptitle2, pcapname3=pcaptitle3,
                  pcapname4=pcaptitle4, title="AIC/BIC position", plotfile=plot_dir + "paper/aic-bic-order-v2")


def paper_aicbic_plots_pt2():
    """

    :return:
    """
    plots_root = "./plots/"
    data_dir = "lan-firewall/"
    plot_dir = "paper/"
    empirical_cdf_crossval = "Empirical CDF function cross-validation.dat"
    pareto_lr_cdf_approximation = "Pareto aproximation (linear regression) vs Original set.dat"
    pareto_mlh_cdf_approximation = "Pareto aproximation (maximum likehood) vs Original set.dat"
    weibull_cdf_approximation = "Weibull aproximation vs Original set.dat"
    plot_lanfirewall_cdfs(plots_root, data_dir, plot_dir, empirical_cdf_crossval, pareto_lr_cdf_approximation,
                          pareto_mlh_cdf_approximation, weibull_cdf_approximation)


########################################################################################################################
# Temporaty functions
########################################################################################################################

def plot_sim_cost_history(plot_dir):
    """

    :param plot_dir:
    :return:
    """
    """
    Temporary function to plot just the cost history
    # parser.add_argument("--costhistory", type=str, nargs="+", help="directory", required=False)
    # elif args["costhistory"]:
    #    # ./plots.py --costhistory "./plots/skype/"
    #    plot_sim_cost_history(args.get("costhistory")[0])
    :param plot_dir:
    :return:
    """
    plot_cost_history(plot_dir=plot_dir, datafile="Weibull - Cost J(iterations) convergence.dat",
                      plot_title="Weibull: Linear Regression Cost History",
                      plot_file="Weibull - Cost J(iterations) convergence")
    plot_cost_history(plot_dir=plot_dir, datafile="Exponential - Cost J(iterations) convergence.dat",
                      plot_title="Exponential: Linear Regression Cost History",
                      plot_file="Exponential - Cost J(iterations) convergence")
    plot_cost_history(plot_dir=plot_dir, datafile="Pareto - Cost J(iterations) convergence.dat",
                      plot_title="Pareto: Linear Regression Cost History",
                      plot_file="Pareto - Cost J(iterations) convergence")
    plot_cost_history(plot_dir=plot_dir, datafile="Cauchy - Cost J(iterations) convergence.dat",
                      plot_title="Cauchy - Cost J(iterations) convergence",
                      plot_file="Cauchy - Cost J(iterations) convergence")


def plot_cauchy_linear_regression(plot_dir):
    """
    Used to plot just the Cauchy Linear regression figures
    # parser.add_argument("--cauchy", type=str, nargs="+", help="directory", required=False)
    # elif args["cauchy"]:
    #    # ./plots.py --costhistory "./plots/skype/"
    #    plot_cauchy_linear_regression(args.get("cauchy")[0])
    :param plot_dir:
    :return:
    """
    plot_linear_regression(plot_dir=plot_dir, datafile="Cauchy - Linearized data and linear fitting.dat",
                           plot_title="Cauchy - Linearized data and linear fitting",
                           plot_file="Cauchy - Linearized data and linear fitting")


########################################################################################################################
# Help tutorial/tests
########################################################################################################################

def help_menu():
    """
    Display a short tutorial of how use this script
    :return:
    """
    print("Usage: plot.py [OPTION] [DIRECTORY]")
    print("Create the plots for the simulations.")
    print("  --help         Display this help menu")
    print("  --simulation   Create the plots using the data created by a  run.py execution.")
    print("                 [DIRECTORY] is the relative path for directory where the data was generated")
    print("                 Eg.:")
    print("                 ./plots.py --simulation \"./plots/skype/\"")
    print("  --paper        Crete the plots for the article. It uses the simulation data on the directories:")
    print("                 * plots/bigFlows: lan gateway pcap")
    print("                 * plots/equinix-1s: wan pcap")
    print("                 * plots/lanDiurnal: ")
    print("                 * plots/skype: Skype pcap")
    print("                 Eg.:")
    print("                 ./plots.py --paper")
    print("")


def run_tests():
    """
    Used to run function under development
    :return:
    """
    plot_dir = "./plots/"
    term.command("mkdir -p " + plot_dir + 'paper/', color="green")
    pcaptitle1 = "skype"
    pcaptitle2 = "lan-gateway"
    pcaptitle3 = "lan-firewall-diurnal"
    pcaptitle4 = "wan"
    costfunction1 = "skype/costFunction.dat"
    costfunction2 = "bigFlows/costFunction.dat"
    costfunction3 = "lanDiurnal/costFunction.dat"
    costfunction4 = "equinix-1s/costFunction.dat"
    aicbic1 = "skype/Aic-Bic.dat.csv"
    aicbic2 = "bigFlows/Aic-Bic.dat.csv"
    aicbic3 = "lanDiurnal/Aic-Bic.dat.csv"
    aicbic4 = "equinix-1s/Aic-Bic.dat.csv"
    plot_cost_function_all2(costfunction1=plot_dir + costfunction1, costfunction2=plot_dir + costfunction2,
                            costfunction3=plot_dir + costfunction3, costfunction4=plot_dir + costfunction4,
                            pcapname1=pcaptitle1, pcapname2=pcaptitle2, pcapname3=pcaptitle3,
                            pcapname4=pcaptitle4, title="Cost Function Sumary",
                            plotfile=plot_dir + "paper/cost-function-summary-v2")
    plot_aic_bic2(aicbicfile1=plot_dir + aicbic1, aicbicfile2=plot_dir + aicbic2,
                  aicbicfile3=plot_dir + aicbic3, aicbicfile4=plot_dir + aicbic4,
                  pcapname1=pcaptitle1, pcapname2=pcaptitle2, pcapname3=pcaptitle3,
                  pcapname4=pcaptitle4, title="AIC/BIC position", plotfile=plot_dir + "paper/aic-bic-order-v2")


########################################################################################################################
# Main
########################################################################################################################

if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(description='Run plotter for simulations or paper plots')
    parser.add_argument("--simulation", type=str, nargs="+",
                        help="run Pcap simulation plots on the directory SIMULATION, using the data created by run.py.",
                        required=False)
    parser.add_argument("--paper", action='store_true',
                        help="run sumarry plots for 4 the 4 experiments used on the paper..",
                        required=False)

    parser.add_argument("--man", action='store_true', help="Manual", required=False)
    parser.add_argument("--test", action='store_true', help="Run plots being developed", required=False)

    args = vars(parser.parse_args())  # convert parser object to a dictionary
    # args = args = {'paper': False, 'test': True, 'simulation': 'plots/skype/', 'test2': None}
    # args = args = {'paper': False, 'paper2': False, 'test': True, 'simulation': 'plots/skype/', 'test2': None}
    # args = {"paper": False, "test": True, "simulation": None, "man": None}
    if args["simulation"]:
        # ./plots.py --simulation "./plots/skype/"
        term.print_color(color="green", data='./plots.py --simulation "{0}"'.format(args.get("simulation")[0]))
        dataprocessor_simulation_plot(args.get('simulation')[0])
    elif args["paper"]:
        # ./plots.py --paper
        paper_aicbic_plots_pt1()
        # > to enable this plot script is required that the dataprocessor daves the cross validation empirical dataset
        # paper_aicbic_plots_pt2()
    elif args["man"]:
        # ./plots.py --help
        help_menu()
    elif args["test"]:
        run_tests()
