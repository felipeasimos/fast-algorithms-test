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

# --- Generate more frequent distinct line styles ---
# Line types and widths
dash_types = 4
widths = "2 3 1.5 2.5"
width_array = words(widths)

# Define some dash patterns
set style line 1 dt solid
set style line 2 dt 2
set style line 3 dt 3
set style line 4 dt 4

do for [i=1:20] {
    dt = (i-1) % dash_types + 1
    lw = word(widths, (i-1) % width_array + 1)
    set style line i lc i dt dt lw lw
}


# --- Plot all columns except first ---
plot for [col=2:*] FILENAME using 1:col with lines ls (col-1) title columnheader(col)

pause -1 "Press Enter or close the window to exit"

