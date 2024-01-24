## StreamSpot Data Parser
For Unicorn's analyzer to work, make sure for each graph,
the node IDs start from 0 and do not skip numbers. If you use `parse_fast.py`, you can provide `-a` flag
and the parser will rearrange the graph's node ID to be Unicorn-compliant.

### Usage

You can now use `parse_fast.py` to parse a single graph at once.
However, you must install `pandas`:
```
pip install pandas
```
Using `parse_fast.py` is highly recommended. Run:
```
python parse_fast.py -h
```
to understand the required arguments.


### Graph Format
The input graph should have one line per edge, and each edge should look like this:
```
4	a	5	c	p	0
```
Where:
* `4`: the node ID of the source node of the edge
* `a`: the type of the source node
* `5`: the node ID of the destination node of the edge
* `c`: the type of the destination node
* `p`: the type of the edge
* `0`: the ID of the graph

With the ID of the graph, you can in fact include multiple graphs in the same input file,
but you must specify the ID when parsing the graph.

The *base* output graph also contains one line per edge, and each edge would look like this:
```
4 5 a:c:p:1
```
The only difference from the format above, besides the format itself, is the logical timestamp (`1` in the example above)
that is attached to each edge.
Note that graph ID is no longer part of the edge, so the output file contains only a single graph as specified during parsing.

The *stream* output graph, written to a different file, again contains one line per edge, and each edge would look like this:
```
4 14 a:c:u:0:1:26
```
`4` and `14` are still the ID of the source and the destination node.
`a`, `c`, and `u` are the type of the source node, the destination node, and the edge, respectively.
`26` is the logical timestamp.
For each edge, `0` means we have observed the same node ID previously while `1` means that this is a new node never before seen.
There two fields are used in the later graph processing framework and may not be useful to you if you use a different framework.


