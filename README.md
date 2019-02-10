# Automated Selection of Network Traffic Models through Bayesian and Akaike Information Criteria

This repository constains the set of scripts used on the paper "Automated Selection of Network Traffic Models
through Bayesian and Akaike Information Criteria", and a tutorial of how to reproduce the experiments with the same or new pcap files.

Sumary:
1. Dependencies
2. Setup Enviroment
3. Tutorial
4. Files documentation

---

##1. Dependencies

To run these scripts, the follow dependencies are required:

###1.1 Python
```bash
sudo apt install python3-pip
pip install -U matplotlib --user
pip3 install numpy
sudo apt-get install python3-tk
```

###1.2 Octave
To install and setup octave, use the follow commands:
```bash
# add repository
sudo add-apt-repository ppa:octave/stable
sudo apt-get update
# install Octave
sudo apt-get install octave
sudo apt-get install liboctave-dev
# other packages dependencies
sudo apt-get install gnuplot epstool transfig pstoedit
```

Install statistic packeges on Octave to run the simulations. Run the follow command to start Octave CLI:
```bash
octave-cli
```

Inside Octave CLI, run the folloe commands:
```command
octave> pkg -forge install io
octave> pkg -forge install statistics
```

After running these commands, a directory on home called .config/octave will appear. But it may have some ownership/access problems. To solve it, run this command on Shell terminal:
```bash
sudo chown $USER ~/.config/octave/qt-settings
```

---
##2. Setup Enviroment


###2.1 Setup

The pcap files used on these experiments are provided on the repository [https://github.com/AndersonPaschoalon/Pcaps](https://github.com/AndersonPaschoalon/Pcaps).
To run the tests, we recomend clonning this repository (or its [code-only version](code.only)) and the Pcap repository side by side:

(root-dir)/
  ├── aic-bic-paper/
  ├── Pcaps/

To prepare the You can do this by using the follow commands:

```bash
mkdir aic-bic-tests
cd aic-bic-tests
git clone https://github.com/AndersonPaschoalon/Pcaps
git clone git clone https://github.com/AndersonPaschoalon/aic-bic-paper
```
To generate the pcap files:
```bash
./git-setup.sh --merge
```
After that, to clean-up the local repository (excludign part files), you may execute:
```bash
./git-setup.sh --rm
```

###2.2 Runing

To run the simulations, use run.py. This is a script to automate and simplify the script execution, maintaining the consistency, without having to know inner details.
Runing `run.py --help` we have an example:
```command
./run.py <pcap_file>  <simulation_name>
	Eg.: ./run.py ../pcaps/wireshark-wiki_http.pcap  wombat
```
The first argument must be the relative path, and the second the name of the simulation. A directory with the simulation nada will be created at plots/ directory. After that, to generate the figures, run:
```command
./plot.py --simulation "plots/<simulation_name>"
```

---
##3. Tutorial

Supose you have a pcap file the directory (relative to this one) `../Pcaps/wombat-test.pcap`. We may script the execution of the tests as below:

```bash
./run.py ../Pcaps/wombat-tests.pcap  wombat
./plot.py --simulation "plots/wombat"
```

The command `./plot.py --paper` will also create some aditional plots for the paper. 
To recreate all the plots on the paper, after creating the enviroment, we must execute:
```bash
./run.py Pcaps/skype.pcap skype
./run.py Pcaps/bigFlows.pcap bigFlows
./run.py Pcaps/equinix-1s.pcap equinix-1s
./run.py Pcaps/lanDiurnal.pcap lanDiurnal
./plots.py --simulation "./plots/skype/"
./plot.py --simulation "./plots/bigFlows/"
./plot.py --simulation "./plots/equinix-1s/"
./plot.py --simulation "./plots/lanDiurnal/"
./plot.py --paper
```

---
##4. Files documentation

### dataProcessor
These are the set of scripts located at the dataProcessor directory, used by the run.py script.  
- __pcap-filter.sh__ : extract inter packet times from pcaps  
    ├── _timerelative2timedelta.m_: script used by pcap filter  
- __dataProcessor.m__: run simulations and stores the data on data/ directory  
    ├── _adiff.m_: calc the absolute difference   
    ├── _cdfCauchyPlot.m_: create the values of a Cauchy CDF  distribution, and plot in a figure  
    ├── _cdfExponentialPlot.m_: create the values of a Exponential CDF  distribution, and plot in a figure  
    ├── _cdfNormalPlot.m_: create the values of a Normal CDF  distribution, and plot in a figure  
    ├── _cdfWeibullPlot.m_: create the values of a Weibull CDF  distribution, and plot in a figure  
    ├── _cdfParetoPlot.m_: create the values of a Pareto CDF  distribution, and plot in a figure  
    ├── _cdfplot.m_: create the values of a Cauchy CDF  distribution, and plot in a figure  
    ├── _computeCost.m_: compute cost for linear regression  
    ├── _cumulativeData.m_: acumulates a vector  
    ├── _gradientDescent.m_: gradient descendent algorithm  
    ├── _informationCriterion.m_:  
    ├── _likehood\_log.m_:  
    ├── _matrix2File.m_: save matrix into a text file  
    ├── _empiricalCdf.m_: eval empirical CDF  
    ├── _plotData.m_: wrapper for plot x and y data  
    ├── _qqPlot.m_: wrapper for qqplots on octave wraper
    ├── _sameLength.m_: ensure two vecters have the same size. If not, the bigget is truncated.
    ├── _setxlabels.m_: set x tick labels on axis on figures  
    ├── _sff2File.m_: vector to file  
    ├── __data/__: place where dataProcessor.m saves the generated data  
    ├── __figures/__: figures plotted by dataProcessor  
- __calcCostFunction.py__: aux script, this script calcs the cost function for the simulated data and saves in the file costFunction.dat.  
- __aicBicRelativeDiff.py__: script to calc the relative difference between AIC and BIC. 







