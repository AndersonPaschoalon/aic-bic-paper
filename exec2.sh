#!/bin/bash

./run.py ./Pcaps/arp-storm.pcap arp-storm
./plot.py --simulation "./plots/arp-storm/"

#./run.py ./Pcaps/skype.pcap skype
#./plot.py --simulation "./plots/skype/"

#./run.py ../../ProjetoMestrado/Tests/pcaps/lanDiurnal.pcap lanDiurnal-02
#./run.py pcap/cmp_IR_sequence_OpenSSL-Cryptlib.pcap openssl
#./run.py ../../ProjetoMestrado/Tests/pcaps/skype.pcap skype
#./plot.py --simulation "./plots/skype/"
#./plot.py --simulation "./plots/bigFlows/"
#./plot.py --simulation "./plots/equinix-1s/"
#./plot.py --simulation "./plots/lanDiurnal/"
# plot just cost history
#./plot.py --costhistory "./plots/skype/"
#./plot.py --costhistory "./plots/bigFlows/"
#./plot.py --costhistory "./plots/equinix-1s/"
#./plot.py --costhistory "./plots/lanDiurnal/"


#./plot.py --cauchy "./plots/skype/"
#./plot.py --cauchy "./plots/bigFlows/"
#./plot.py --cauchy "./plots/equinix-1s/"
#./plot.py --cauchy "./plots/lanDiurnal/"



        
