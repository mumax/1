plot "stdprobl4-euler/datatable.txt" using 1:3 with lines, "solution3.txt" using ($1*1E-9):9 with lines;