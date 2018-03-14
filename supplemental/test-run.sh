# change the paths as you like
cd ../release
./reduction-grid $((1*1048576)) $((1024*1048576)) > ../results/K80/cuda-9.0.176/reduction-grid-823mhz.csv
./reduction-mono $((1*1048576)) $((1024*1048576)) > ../results/K80/cuda-9.0.176/reduction-mono-823mhz.csv
./saxpy-grid $((1*1048576)) $((1024*1048576)) > ../results/K80/cuda-9.0.176/saxpy-grid-823mhz.csv
./saxpy-mono $((1*1048576)) $((1024*1048576)) > ../results/K80/cuda-9.0.176/saxpy-mono-823mhz.csv
