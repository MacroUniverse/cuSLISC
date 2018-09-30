# Makefile

exe = main.x

source = main.cu nr3plus.cu
objects = $(source:.cu=.o)
compiler = nvcc
flags =  -arch=sm_60 -g -G -std=c++11
# -g -G -O3

$(exe):$(objects)
	$(compiler) -o $(exe) $(flags) $(objects)

$(objects):$(source)
	$(compiler) -c $(flags) $(source)

# clean all except source
clean:
	rm -f *.o *.x *.gch
