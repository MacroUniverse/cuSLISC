# Makefile

exe = main.x
gputype = CUSLISC_GTX1080

source = main.cu nr3plus.cu cusliscplus.cu
objects = $(source:.cu=.o)
compiler = nvcc
flags =  -arch=sm_60 -g -G -std=c++11
# -g -G -O3

$(exe):$(objects)
	$(compiler) -o $(exe) $(flags) $(objects)

$(objects):$(source)
	$(compiler) -c $(flags) $(source) -D $(gputype)

# clean all except source
clean:
	rm -f *.o *.x *.gch
