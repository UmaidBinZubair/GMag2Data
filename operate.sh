#!/bin/bash
## declare an array variable
declare -a array=("2004" "2005" "2006" "2007" "2008" "2009" "2010" "2011" "2012" "2013" "2014" "2015" "2016" "2017" "2019" "2020")

# get length of an array
arraylength=${#array[@]}

# use for loop to read all values and indexes
#for (( i=2010; i<2011+1; i++ ));
for i in ${array[@]}
do
 python mag2data.py -i processed_books/$i/pages/ -o new_processed_books/$i -f $i.xlsx -y $i
# python pdfreader.py -r Books/$i.pdf -o new_processed_books/$i/ -i processed_books/$i/ -n $i
done
