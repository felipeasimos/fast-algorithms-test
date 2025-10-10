#!/usr/bin/env gnuplot

FILENAME = "plot.csv"
TITLE = "GEMM"

set terminal qt persist noenhanced noraise font "Sans,12" size 1000,600 title TITLE
set grid
set style data lines
set key autotitle columnheader
set datafile separator ","
set autoscale fix

# --- Extract X-axis label from first column header ---
xheader = system("head -n 1 ".FILENAME." | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, \"\", $1); gsub(/^\"|\"$/, \"\", $1); print $1}'")
set xlabel xheader

# --- Plot all columns except first ---
plot for [col=2:*] FILENAME using 1:col title columnheader(col)

pause -1 "Press Enter or close the window to exit"

