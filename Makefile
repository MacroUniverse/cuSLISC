# Makefile

exe = main.x
gputype = -D CUSLISC_GTX1080
debug = -D _CHECKSETSYS_ -D _CHECKBOUND_ -D _CHECKTYPE_

cusource = main.cu cusliscplus.cu
cppsource = nr3plus.cpp
source = $(cusource) $(cppsource)
objects = $(cusource:.cu=.o) $(cppsource:.cpp=.o)
compiler = nvcc
flags =  -arch=sm_60 -g -G -std=c++11
# -g -G -O3

$(exe):$(objects)
	$(compiler) -o $(exe) $(flags) $(objects)

$(objects):$(source)
	$(compiler) -c $(flags) $(source) -D _CUSLISC_ $(gputype) $(debug)

# clean all except source
clean:
	rm -f *.o *.x *.gch
