# Makefile

exe = main.x
gputype = -D CUSLISC_GTX1080
debug = -D _CHECKSETSYS_ -D _CHECKBOUND_

source = main.cu nr3plus.cu cusliscplus.cu
objects = $(source:.cu=.o)
compiler = nvcc
flags =  -arch=sm_60 -g -G -std=c++11
# -g -G -O3

$(exe):$(objects)
	$(compiler) -o $(exe) $(flags) $(objects)

$(objects):$(source)
	$(compiler) -c $(flags) $(source) $(gputype) $(debug)

# clean all except source
clean:
	rm -f *.o *.x *.gch
