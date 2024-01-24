# GraphChi for Unicorn
Unicorn uses GraphChi to perform vertex-centric graph processing.
The input graph should be in the format output from one of the [parsers](https://github.com/crimson-unicorn/parsers).
The code specific for Unicorn is in this directory (i.e., `analyzer/unicorn`),
but you should use the `Makefile` in the parent directory (i.e., `analyzer/`) to compile and run the code.

## Compile
Unicorn uses a number of `gcc` preprocessor macros:
* `-DSKETCH_SIZE=<SIZE>`: (required) defines the size of the graph sketch
* `-DK_HOPS=<HOP>`: (required) defines the size of the neighborhood for each vertex to explore
* `-DMEMORY -DPREGEN=<NUM>`: (optional, recommended) pre-samples `NUM` of random variables for hashing and stores them in memory
* `-DUSEWINDOW`: (optional) uses `window` argument (described below) to determine the frequency of sketch generation
* `-DBASESKETCH`: (optional) uses the base graph sketch as the first sketch; this macro is recommended if `-DUSEWINDOW` is set. **DO NOT SET THIS FLAG IF -DUSEWINDOW IS NOT SET:** the first sketch is already from the base graph if `-DUSEWINDOW` is not set; you will end up with two sketches describing the base graph
* `-DDEBUG`: (optional) runs in debug mode with verbose output
* `-DVIZ`: (optional) writes histogram to files for analysis and histogram visualization

You should set them to compile the code. For example, in `Makefile`, we have:
```
swdebug: CPPFLAGS += -DSKETCH_SIZE=2000 -DK_HOPS=3 -DMEMORY -DPREGEN=10000 -DUSEWINDOW -DBASESKETCH -DDEBUG -g
swdebug: unicorn/main
```
We set the size of the sketch to be 2,000, explore 3-hop neighborhoods, and pre-sample 10,000 random variable sets.
We also use `window` argument to determine the frequency of sketch generation and we record the first sketch from the base graph.
We run the code in debug mode.
You should use this as the template, simply modify the macros as needed, and run:
```
make swdebug
```

## Run
Once you have compiled the code, you can use the following command template and run the code (from the `graphchi-cpp/` directory):
```
bin/unicorn/main filetype edgelist [niters <MAX_NUMBER_OF_ITERATIONS>] base <BASE_GRAPH_FILE_PATH> stream <STREAM_GRAPH_FILE_PATH> [decay <DECAY_FREQUENCY>] [lambda <DECAY_RATE]> [window <WINDOW_SIZE>] [batch <BATCH_SIZE>] [chunkify <1_OR_0>] [chunk_size <SIZE>] sketch <GRAPH_SKETCH_FILE_PATH> [histogram <HISTOGRAM_FILE_PREFIX_NAME>]
```
* `filetype`: must be `edgelist`. *Do not change this argument value*
* `niters`: (optional) the maximum number of iterations to analyze the streaming graph. You can set this value as big as possible, but Unicorn will stop once the entire graph has been processed. The default value (which is set to be 1,000,000) is big enough, so in most cases, you do *not* need to set this value
* `base`: (required) the file path to the base graph
* `stream`: (required) the file path to the streaming part of the graph
* `decay`: (optional) the number of vertices we have processed in the streaming part of the graph before we perform the gradually forgetting scheme on the graph histogram. The default value (which is set to be 10) is likely *not* what you want. *You are strongly recommended to set this value suitable for your application*
* `lambda`: (optional) the rate of the gradually forgetting scheme. This is used with the `decay` option. The default value (which is set to be `0.02`) may work for you. You can set any value between 0 and 1
* `window`: (optional) the number of vertices we have processed in the streaming part of the graph before we record a new graph sketch. This frequency is used if `USEWINDOW` preprocessor macro is set; otherwise we will use `batch` to set the frequency. Even if you set `USEWINDOW`, you are not required to set this value since we provide a default value, which is 500. However, the default value is likely *not* what you want. *We strongly recommend you to set this value suitable for your application*
* `batch`: (optional) the number of streaming edges batched together to update the graph. If `USEWINDOW` is *not* set, this is also the frequency we use to record sketches. That is, we will stream `BATCH_SIZE` edges to the graph, run our algorithm to update all the vertices, the histogram, and the sketch, and then record the sketch. If you use this value as the frequency, *we recommend that you have the base graph the same size as* `BATCH_SIZE`. Please refer to the documentation in [parsers](https://github.com/crimson-unicorn/parsers) to understand how you can set the base graph size. If you follow our recommendation, each sketch will include the same (i.e., `BATCH_SIZE`) number of additional edges
* `chunkify`: (optional) if you want to chunk the labels. You can set it to be either 1 (chunk) or 0 (do not chunk); the default is 1
* `chunk_size`: (optional) if you set `chunkify` to 1, you should set the size of each chunk (the default is 5, which may or may not work for you)
* `sketch`: (required) the file path to graph sketches
* `histogram`: (optional) you must provide the prefix name for file paths to all histogram files *if and only if* the `VIZ` macro is set. Unicorn wil generate one histogram file per sketch generation; that is, the number of sketches in the sketch file is the same as the number of histogram files. *Do not provide this argument if the macro is not set*
