#!/bin/bash

#./run.py ../../ProjetoMestrado/Tests/pcaps/lanDiurnal.pcap lanDiurnal-02
#./run.py pcap/cmp_IR_sequence_OpenSSL-Cryptlib.pcap openssl
#./run.py ../../ProjetoMestrado/Tests/pcaps/skype.pcap skype
#./plots.py --simulation "./plots/skype/"
./plot.py --simulation "./plots/bigFlows/"
./plot.py --simulation "./plots/equinix-1s/"
./plot.py --simulation "./plots/lanDiurnal/"


# plot just cost history
./plot.py --costhistory "./plots/skype/"


        
