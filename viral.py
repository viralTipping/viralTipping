"""
Viral Marketing Library for Python.

References:
This library references "A Scalable Heuristic for Viral Marketing Under
the Tipping Model," by the authors. 
The paper may be found at http://arxiv.org/abs/1309.2963

This project is licensed under the MIT License (MIT):

Copyright (c) 2013 P. Shakarian, S. Eyre, D. Paulo.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Authors:
P. Shakarian
E-mail: paulo@shakarian.net

S. Eyre
E-mail: sean.k.eyre@gmail.com

15 December 2013
"""
#Dependencies:
import numpy, networkx, BinomialHeap, time, math, copy, os, sys, csv, scipy
from scipy import random
from random import choice

#Constants:
infinity = 1000000000000000000

#Functions:

def findSeedSet(network, threshold, linearFlag=True, directed=True, verbose=False):
	"""
	Function that accepts a directed NetworkX network 
	(with the option for undirected networks) and an integer 
	or fractional threshold and returns a seed set. Set verbose=True if you wish to see benchmarks while conducting tests. 
	IN:
	1. NetworkX network
	2. Integer Threshold
	3. (Optional) Boolean Flag describing if it is an integer or fractional threshold (default True, linear)
	4. (Optional) Directional Flag (default True)
	5. (Optional) Verbose Flag (default False)
	OUT:
	1. Set of nodes that activates the entire network

	TO-DO:
	1. Manage undirected networks
	"""
	#Set up the variables
	#made a make a boolean flag for each node
	nodeBooleanDictionary = createBooleanDictionary(network)
	
	if verbose:
		print("[.] Executing setup tasks:")
	
	startTime = time.clock() #start the timer
		
	numberOfNodes = network.number_of_nodes() #save the number of nodes
	numberOfEdges = len(network.edges()) #save the number of edges
	
	nFact = float(numberOfNodes - 1) #set nFact at 1 less than the number of nodes in the network
	#make a dictionary of nodes to their k-value
	if directed:
		kValues = networkx.algorithms.centrality.in_degree_centrality(network)
	else:
		kValues = networkx.algorithms.centrality.degree_centrality(network)
	verticeList = list(kValues.keys()) #save a list of vertices 
	
	pointerDictionary = {} #make a dictionary to hold pointers in
	valueDictionary = {} #make a dictionary to hold values in 
	heap = BinomialHeap.heap() #make a new binomial heap
	
	#for every node/vertice, set each k-value:
	for item in verticeList:
		kValues[item] = round(kValues[item]*nFact)
		currentValue = kValues[item]
		if linearFlag: #if it's a linear threshold propogation
			if kValues[item] > threshold: #if the k-value for this node is greater than the threshold
				kValues[item] = threshold # set the k-value to the threshold
		else: #if it's a fractional threshold
			if not(threshold == 1.0):
				#if the threshold isn't 1.0, set the value to in-degree*threshold (k = d_in * theta)
				kValues[item] = math.ceil(kValues[item]*threshold)
			#else: we're multiplying by 1, so don't we do any work
		distance = currentValue - kValues[item] #set the distance to d_in - k (ie. distance = d_in - k)
		#add the node and value to the binomial heap, then store a pointer for it
		pointerDictionary[item] = heap.insert(distance, item)
		kValues[item] = distance #update k from d_in to the distance

	time2 = time.clock()
	setupTime = time2 - startTime
	if verbose:
		print("[..] Setup tasks completed ("+str(setupTime)+" sec).")
		print("[.] Running decompostition algorithm.")
	
	#run the main loop of decomposition
	time3 = time.clock()
	result = decompositionLoop(kValues, pointerDictionary, heap, network, nodeBooleanDictionary)
	time4 = time.clock()
	
	numerator = float(len(result))
	percentResult = 100*numerator/float(nFact+1) #what percent of the network the result comprises
	algorithmTime = time4-time3
	totalTime = algorithmTime+setupTime
	if verbose:
		print("[..] Decomposition algorithm completed ("+str(algorithmTime)+" sec).")
		print("[..] The seed is "+ str(percentResult)+"% of the entire network.")
	
	return result


def findLinearSeedSet(network, threshold, directed=True, verbose=False):
	"""
	Function that accepts a directed NetworkX network 
	(with the option for undirected networks) and an integer threshold
	and returns a seed set. Set verbose=True if you wish to see benchmarks while conducting tests.
	IN:
	1. NetworkX network
	2. Integer Threshold
	3. (Optional) Directional Flag (default True)
	4. (Optional) Verbose Flag (default False)
	OUT:
	1. Set of nodes that activates the entire network
	"""
	return findSeedSet(network, threshold, True, directed, verbose)

def findFractionalSeedSet(network, threshold, directed=True, verbose=False):
	"""
	Function that accepts a directed NetworkX network
	(with option for undirected) and a fractional threshold
	and returns a seed set. Set verbose=True if you wish to see benchmarks while conducting tests.
	IN:
	1. NetworkX network
	2. Fractional Threshold
	3. (Optional) Directional Flag (default True)
	4. (Optional) Verbose Flag (default False)
	OUT:
	1. Set of nodes that activates the entire network
	"""
	return findSeedSet(network, threshold, False, directed, verbose)


def findActivation(network, seedSet, threshold, linearFlag=True, directed=True, verbose=False):
	"""
	General function that accepts a directed NetworkX network
	(with option for undirected), set of initial nodes and an 
	integer threshold or fractional threshold and returns who is infected.
	IN: 
	1. NetworkX network
	2. Set of seed nodes
	3. Threshold
	4. (Optional) Boolean Flag describing if it is an integer or fractional threshold (default True, linear)
	5. (Optional) Directional Flag (default True) 
	6. (Optional) Verbose Flag (default False)
	OUT:
	1. Set of activated nodes
	"""
	#initialize a dictionary for k-values at the in-degree for each node
	if directed:
		kValues = networkx.algorithms.centrality.in_degree_centrality(network)
	else:
		kValues = networkx.algorithms.centrality.degree_centrality(network)
	nFact = float(network.number_of_nodes()-1) #set nFact as the number of nodes in the network - 1
	verticeList = list(kValues.keys())
	
	for vertice in verticeList:
		kValues[vertice] = round(kValues[vertice]*nFact)
		if linearFlag:
			if kValues[vertice] > threshold: #if the k-Value exceeds the threshold
				kValues[vertice] = threshold #set the k-Value equal to the threshold
			#else: the degree is less than the threshold, so should that node's threshold is its degree
		else:
			if not(threshold == 1.0):
				kValues[vertice] = math.ceil(kValues[vertice]*threshold)
	result = findActivationOuterLoop(network, seedSet, kValues, directed)
	return result

def findLinearActivation(network, seedSet, threshold, directed=True, verbose=False):
	"""
	Function that accepts a directed NetworkX network
	(with option for undirected), set of initial nodes and an 
	integer threshold and returns who is infected.
	IN: 
	1. NetworkX network
	2. Set of seed nodes
	3. Threshold integer
	4. (Optional) Directional Flag (default True) 
	5. (Optional) Verbose Flag (default False)
	OUT:
	1. Set of activated nodes
	"""
	findActivation(network, seedSet, threshold, True, directed, verbose)
	

def findFractionalActivation(network, seedSet, threshold, directed=True, verbose=False):
	"""
	Function that accepts a directed NetworkX network 
	(with option for undirected) set of intitial nodes and a 
	fractional threshold and returns who is infected.
	IN: 
	1. NetworkX network
	2. Set of seed nodes
	3. Threshold floating point number
	4. (Optional) Directional Flag (default True) 
	5. (Optional) Verbose Flag (default False)
	OUT:
	1. Set of activated nodes
	"""
	findActivation(network, seedSet, threshold, False, directed, verbose)


def createBooleanDictionary(network):
	"""
	Function to create a boolean flag for every node in a network
	IN:
	1. NetworkX network
	Out:
	1. Dictionary with nodes as keys and boolean flags as values
	"""
	result = {}
	for item in network.nodes_iter():
		result[item] = True
	return result


def heap2set(heap):
	"""
	Function to turn a binomial heap (as implemented in dependent/imported library)
	into a Python set
	IN:
	1. Binomial Heap
	Out:
	1. Set
	"""
	result = set()
	for node in heap:
		result.add(node)
	return result

	
def decrimentNode(node, valueDictionary, pointerDictionary, heap):
	"""
	Function to decriment the value of a node's distance and then reflect that change 
	in the collection of pointers and the binomial heap
	IN:
	1. Node
	2. Dictionary of distance values
	3. Dictionary of pointers referencing nodes in the binomial heap
	4. The binomial heap
	Out:
	1. Nothing - mutates its inputs
	"""
	if not (valueDictionary[node] == infinity):
		valueDictionary[node]=valueDictionary[node]-1 #decriment the distance stored in the dictionary by 1
		if valueDictionary[node] < 0: #if the result is now less than 0
			valueDictionary[node] = infinity #make the distance to the node infinity
			pointerDictionary[node].delete() #remove it from the heap
			pointerDictionary[node]=heap.insert(valueDictionary[node], node) #replace it in the heap with the new value
		else: 
			pointerDictionary[node].decrease(valueDictionary[node]) #set the nodes new value

		
def decrimentOutNeighbors(node, valueDictionary, pointerDictionary, heap, network, nodeBooleanDictionary):
	"""
	Function to decriment all of a nodes neighbors.
	IN:
	1. Node
	2. Dictionary of distance values
	3. Dictionary of pointers referencing nodes in the binomial heap
	4. The binomial heap
	5. NetworkX network
	6. Dictionary of flags for each node
	Out:
	1. Nothing - mutates its inputs
	"""	
	for neighbor in network.neighbors_iter(node): #Neighbors_iter will work for DiGraphs and Graphs and is the same as successors_iter
	#  Ref (http://networkx.github.io/documentation/latest/reference/generated/networkx.DiGraph.successors_iter.html?highlight=successors#networkx.DiGraph.successors_iter)
		if nodeBooleanDictionary[neighbor]: #if the neighbor is still in the network
			decrimentNode(neighbor, valueDictionary, pointerDictionary, heap)

	
def pickMinimumAndRemember(valueDictionary, pointerDictionary, heap, network, nodeBooleanDictionary):
	"""
	Function to pick the node with the smallest distance in the heap then mutate its neighbors
	by decrimenting all of their distance values. Returns true when all nodes have been examined (Does not necessarily
	mean that the heap is empty- nodes that have a value of infinity will not be examined but are still in the heap).
	IN:
	1. Dictionary of distance values
	2. Dictionary of pointers referencing nodes in the binomial heap
	3. The binomial heap
	4. NetworkX network
	5. Dictionary of flags for each node
	Out:
	1. Boolean describing if this is the last node in the heap or not
	"""
	lastEntry = False
	topNode = heap.min() #store the node with the smallest distance to activation
	topValue = valueDictionary[topNode]
	if not (topValue == infinity): #"if this node can still be activated"
		heap.extract_min() #remove the top node from the heap
		decrimentOutNeighbors(topNode, valueDictionary, pointerDictionary, heap, network, nodeBooleanDictionary)
		nodeBooleanDictionary[topNode] = False #mark that the node is no longer in the network
	else:
		lastEntry = True
	return lastEntry
	

def decompositionLoop(valueDictionary, pointerDictionary, heap, network, nodeBooleanDictionary):
	"""
	Function to progress through the network and selectively prune nodes with the smallest distance to activation. At termination,
	all nodes remaining in the heap will have a value of infinity (see constants) and will form the seed set.
	IN:
	1. Dictionary of distance values
	2. Dictionary of pointers referencing nodes in the binomial heap
	3. The binomial heap
	4. NetworkX network
	5. Dictionary of flags for each node
	Out:
	1. A set of nodes remaining in the heap (seed set)
	"""
	stopRunning = False
	while not(stopRunning):
		stopRunning = pickMinimumAndRemember(valueDictionary, pointerDictionary, heap, network, nodeBooleanDictionary)
	return heap2set(heap)

def countActiveNeighbors(vertice, network, seedSet, directed=True):
	"""
	Function that accepts a vertice, NetworkX network, 
	and a set of activated nodes and returns the number of its neighbors that
	have been activated
	IN: 
	1. vertice in the network
	2. NetworkX network
	3. Set of seed nodes
	OUT:
	1. number of neighbors that have been activated
	"""

	result = 0
	if directed:
		for neighbor in network.predecessors_iter(vertice):
			if (neighbor in seedSet):
				result = result + 1
	else:
		for neighbor in network.neighbors_iter(vertice):
			if (neighbor in seedSet):
				result = result + 1
	return float(result)

def meetThreshold(vertice, network, seedSet, workingSet, kValues, directed=True):
	"""
	Function that accepts a vertice, NetworkX network, seed set, 
	set of activated nodes and a dictionary of k-values and 
	returns if the threshold has been surpassed (ie, this vertice has tipped)
	IN: 
	1. vertice in the network
	1. NetworkX network
	2. Set of seed nodes
	3. Set of activated nodes
	4. Dictionary of k-values
	OUT:
	1. Boolean of whether or not vertice was activated
	"""	

	threshold = float(kValues[vertice])
	flag = False
	x = countActiveNeighbors(vertice, network, seedSet, directed)
	if x >= threshold:
		workingSet.add(vertice)
		flag = True
	return flag

def findActivationInnerLoop(network, seedSet, kValues, directed=True):
	"""
	Function that accepts a NetworkX network, set of activated nodes and a dictionary
	of k-values and returns if the propagation spread at this time step
	IN: 
	1. NetworkX network
	2. Set of seed nodes (will be mutated)
	3. Dictionary of k-values
	OUT:
	1. Boolean of whether or not propagation has spread
	2. The mutated seed set as it has propagated 
	"""
	
	change = False
	currentFlag = False
	workingSet = seedSet.copy()
	for vertice in network.nodes_iter(data=False):
		if (vertice not in seedSet):
			currentFlag = meetThreshold(vertice, network, seedSet, workingSet, kValues, directed)
			if currentFlag:
				change = True
	seedSet = workingSet.copy() #mutate the seedSet
	return (change, seedSet)

def findActivationOuterLoop(network, seedSet, kValues, directed=True, verbose=False):
	"""
	Function that accepts a NetworkX network, set of initial nodes and a dictionary
	of k-values and returns who is infected.
	IN: 
	1. NetworkX network
	2. Set of seed nodes
	3. Dictionary of k-values
	4. (Optional) Directed flag (default True)
	5. (Optional) Verbose Flag (default False)
	OUT:
	1. Set of activated nodes
	"""
	
	if verbose:
		print ('[...]Running Outer-Loop')
	change = True
	#while there is still propagation
	while change:
		if verbose:
			print ('Nodes activated: ' + str(len(seedSet))),
			print "-",
			print seedSet
		change, seedSet = findActivationInnerLoop(network, seedSet, kValues, directed)
	return seedSet #at this point, the seed set has mutated to reflect how much the change has propagated 

def readEdgeList(inputFile, delimeter = ',', directed = True):
	"""
	Function that reads in an edge list and file delimeter and returns a NetworkX Graph or DiGraph
	IN: 
	1. text based file name
	2. character delimeter (default ',')
	2. (Optional) Directed flag (default True)
	OUT:
	1. If Directed, returns a DiGraph. Else, returns a Graph.
	"""

	inFile = open(inputFile, 'r')
	if directed:
		graph = networkx.DiGraph()
	else:
		graph = networkx.Graph()
	#for every line in the file
	for line in inFile:
		line = line.strip('\n ()')
		#split the line based on the delimeter
		parts = line.split(delimeter)
		#add an edge from the first part to the second
		graph.add_edge(parts[0].strip(), parts[1].strip())
	inFile.close()
	return graph
	
def readSeedList(inputFile):
	"""
	Function that reads in text file of node names 
	and returns a seed set
	IN: 
	1. text based file name
	OUT:
	1. set of seed nodes.
	"""

	inFile = open(inputFile, 'r')
	result = set()
	for line in inFile:
		result.add(line.strip(' \n\t'))
	inFile.close()
	return result

"""
The Functions nw(g), q(g), and qFast(g, nw) are each helper 
functions which memoize the data necessary to 
construct Combinatorial Local Centrality
"""

def nw(graph):
	"""
	Function that takes in a graph and returns the how many 1st 
	and 2nd degree neighbors each node has
	IN:
	1. a networkX graph
	OUT:
	1. Dictionary of number of nodes within 2-degrees of each node
	"""
	
	nw1 = {}
	for i in graph.nodes():
		nw1[i] = 0 #initialize all nodes in nw1 to value 0
		friends = []
		for j in graph.neighbors(i):
			if j not in friends:
				friends.append(j) #if j is not in the list of friends, add it
			for k in g.neighbors(j): #examine 2 degrees of separation 
				if k not in friends and k != i: #if k is not a mutual friend and is not i
					friends.append(k)
		nw1[i] = len(friends) #set the value for node i to the number of friends within 2 degrees
	return nw1
	
def q(graph):
	"""
	Function that takes in a graph and returns the how many 2nd 
	and 3rd degree neighbors each node has
	IN:
	1. a networkX graph
	OUT:
	1. Dictionary of number of nodes between 2- and 3-degrees of each node
	"""

	g1 = {}
	nw = {}
	nwj = 0
	for i in graph.nodes(): #initialize q1 and nw
		q1[i] = 0
		nw[i] = 0
	for i in graph.nodes():
		for j in graph.neighbors(i):
			friends = []
			for l in graph.neighbors(j):
				if l not in friens:
					friend.append(l)
				for m in graph.neighbors(l):
					if m not in friends and m != j:
						friends.append(m)
			nwj = len(friends)
			q1[i] = q1[i] + nwj
	return q1
	

def qFast(graph, nw):
	"""
	Function that takes in a graph and returns the how many 2nd 
	and 3rd degree neighbors each node has
	IN:
	1. a networkX graph
	2. Dictionary of number of nodes within 2-degrees of each node 
	OUT:
	1. Dictionary of number of nodes between 2- and 3-degrees of each node
	"""

	q1 = {}
	for i in graph.nodes():
		q1[i] = 0
		for j in graph.neighbors(i):
			q1[i] = q1[i] + nw[j]
	return q1

def clcFast(graph, vertices, q):
	"""
	In: Graph, indices of a set of vertices, dictionary q (returned by Q or Q_Fast)
		-returns the Combinatorial Local centrality of the set of vertices indicated on the provided graph
	"""

	clc = 0
	fNeighbors = []
	currentNeighbors = []
	for i in vertices:
		currentNeighbors = graph.neighbors(i)
		for k in currentNeighbors:
			if k not in fNeighbors:
				fNeighbors.append(k)
				clc = clc + q[k]
	return clc

def localCentrality(graph, q):
	"""
	In: graph, dictionary q (returned by Q or Q_Fast)
		-returns a dictionary of local centralities for each node (maps node index -> local centrality)
		-this just leverages CLC_Fast on each node, one at a time, to populate the dictionary
		-if you just need the local centrality of one node quickly, use CLC_Fast(graph, node index, q)
	"""

	localCent = {}
	currentNode = []
	for i in graph.nodes():
		currentNode = [i]
		localCent[i] = clcFast(graph, currentNode, q)
	return localCent
"""
The following are helper functions for altGreedyCLC
"""
def fnGrow(graph, v, fn):
	fNeighbors = list(fn)
	for i in v:
		if (i < 0):
			return list(fn)
		for k in graph.neighbors(i):
			if k not in fNeighbors:
				fNeighbors.append(k)

def altCLC(fn, q):
	clc = 0
	for i in fn:
		clc = clc + q[i]
	return clc

def greedyCLC(graph, k):
	"""
	In: graph, set size k
		-returns the approximate maximum Combinatorial Local Centrality set of nodes, of size k (or less)
	"""
	q = qFast(graph, nw(graph))
	v = []
	vTemp = []
	lastValue = {}
	for i in graph.nodes():
		lastValue[i] = -1
	
	currentValue = 0
	bestInd = -1
	clcValueCurrent = 0
	fn = []
	fnTemp = []
	
	while len(v) < k:
		vTemp = [bestInd]
		fn = fnGrow(graph, vTemp, fn) #grow fn from bestInd
		clcValueCurrent = altCLC(fn, q) #calculate CLC twice
		
		bestValue = 0
		bestInd = -1
		
		for i in graph.nodes():
			if i not in v:
				if lastVal[i] > bestVal or lastVal < 0:
					vTemp = [i]
					fnTemp = fnGrow(graph, vTemp, fn)
					currentValue = altCLC(fnTemp, q) - clcValueCurrent
					lastValue[i] = currentValue
					if currentValue > bestValue:
						bestValue = currentValue
						bestInd = i
		if bestInd > -1:
			v.append(bestInd)
		else:
			break
	return v

def chenSIR(graph, iIn, y, runs):
	"""
	In: graph,indices of an initial set of infected nodes,recovery parameter (usually ~3),total number of simulation runs
		-returns the average number of infected nodes under the SIR Model specified in the
		-chen 12 paper "Identifying influential nodes in complex networks"
		-the recovery parameter is the average number of steps until a node
	"""
	sumI = 0.0
	n = []
	for x in range(runs):
		i = []
		for restock in iIn:
			i.append(restock)
		r = []
		while len(i) > 0:
			infect = -1
			newI = []
			newR = []
			for j in i:
				if random.random() < (1.0/y):
					newR.append(j)
				else:
					n = graph.neighborss(j)
					for e in r:
						if e in n:
							n.remove(e)
					infect = int(len(n)*random.random())
					if len(n) > 0:
						if n[infect] not in i:
							newI.append(n[infect])
			for l in newR:
				i.remove(l)
				r.append(l)
			for k in newI:
				if k not in i:
					i.append(k)
		sumI = sumI + (len(r)*1.0)
	return (sumI/(runs*1.0))
	

def main(arguments):
	#Usage: 
	# -h	return help text
	# <filename>  -thresh=<int|float> -fxn=<seed|act*> [*-seed=<filename>][-del=<char> | -dir=<bool> | -lin=<bool>]
	# Delimeters:
	# 	t	tab delimeted 
	# 	,	comma separated
	if arguments[0] == '-h':
		print("""[.] Usage: 
[..] -h	return help text
[..] <filename> -thresh=<int|float> -fxn=<seed|act*> [*-seed=<filename>]
[-del=<char> | -dir=<bool> | -lin=<bool>]
[.] Delimeters:
[..]	t	tab delimeted 
[..]	,	comma separated (default)""")
	else:
		filename = arguments[0]
		parameters = dict()
		for flag in arguments[1:]:
			parts = flag.strip('-').upper().split('=')
			parameters[parts[0]] = parts[1]
			
		#Set up variables for function calls
		if ('DEL' in parameters) and (parameters['DEL'] == 'T'):
			delimeter = '\t'
		else:
			delimeter = ','
		
		if ('DIR' in parameters) and (parameters['DIR'] == 'FALSE'):
			directed = False
		else:
			directed = True
		
		if ('LIN' in parameters) and (parameters['LIN'] == 'FALSE'):
			linearFlag = False
		else:
			linearFlag = True
		
		if ('THRESH' in parameters):
			if linearFlag: #if linear, cast as integer
				threshold = int(parameters['THRESH'] )
			else:
				threshold = float(parameters['THRESH'])
		else:
			print ("[!]You must specify a threshold.")
			return
		
		#execute function calls
		if 'FXN' in parameters:
			if parameters['FXN'] == 'SEED':
				print ("[.]Reading edge list")
				graph = readEdgeList(filename, delimeter, directed)
				# print graph.nodes()
				print ("[.]Executing function")
				#printing a string for testing purposes
				for n in findSeedSet(graph, threshold, linearFlag, directed):
					print n
			elif parameters['FXN'] == 'ACT':
				#Must include a seed set
				if ('SEED' in parameters):
					print ("[.]Reading seed list")
					seedSet = readSeedList(parameters['SEED'])
					print ("[.]Reading edge list")
					graph = readEdgeList(filename, delimeter, directed)
					print ("[.]Executing function")
					print ("[..]" + str(len(findActivation(graph, seedSet, threshold, linearFlag, directed))) + " of " + str(len(graph.nodes())) + " nodes activated.")
				else:
					print ("[!]Calculating activation requires a seed set")
			else:
				print ("""[!]Unrecognized function.
[.]Usage:
[..]	act		activation
[..]	seed	find seed set""")
				return
	
	
	
if __name__ == "__main__":
	print("[Start]")
	main(sys.argv[1:])
	input = raw_input("[End] Press enter to continue")
