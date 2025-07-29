set style data dots
set nokey
set xrange [0: 6.20439]
set yrange [  6.25308 : 17.42762]
set arrow from  1.59199,   6.25308 to  1.59199,  17.42762 nohead
set arrow from  3.09287,   6.25308 to  3.09287,  17.42762 nohead
set arrow from  3.84331,   6.25308 to  3.84331,  17.42762 nohead
set arrow from  4.90459,   6.25308 to  4.90459,  17.42762 nohead
set xtics ("K"  0.00000,"G"  1.59199,"X"  3.09287,"W"  3.84331,"L"  4.90459,"G"  6.20439)
 plot "nio_band.dat"
