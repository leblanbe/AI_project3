Utility Driven Search for Round Trip Road Trips

1) You will create a local and team-specific copy of the first two sheets of locations and edges for Program 1 . Copy the first two sheets  into csv files, EdgeFile and LocFile, . Ignore the last two sheets for this assignment. See the section Background below for an explanation of edges and locations.

2) You will augment each location and edge with its own preference value. For purposes of testing on Program 1 do the following.

a) Write a function called location_preference_assignments (a, b) that assigns random values between a=0 and b=1 inclusive using a uniform distribution to each location independently.

b) Write a function edge_preference_assignments (a, b) that assigns random values between a=0 and b=0.1 inclusive using a uniform distribution to each edge independently. Note that edges have a smaller preference upper bound than locations for Program 1.

3) Write a function total_preference (roadtrip) that returns the sum of all location and all edge preferences in a road trip. The roadtrip argument can be any graph of valid locations and edges – it need not be round trip, and it need not be connected — this is because the function can be called on partially constructed road-trips at intermediate points in search, as well as being called to evaluate the utility of a fully-connected round-trip. You decide on the internal representation of the roadtrip argument.

4) Write a function called time_estimate (roadtrip, x) that computes the time required by a road trip in terms of its constituent edges and locations as follows.

a) The time at a location is a function of the preference value, vloc, assigned to the location – the greater the preference for a location, the more time spent at the location. Write a function time_at_location (vloc) of your choice, under the constraint that if vloc1 >= vloc2, then tloc1 >= tloc2.

b) The base time required to traverse an edge is time t = d/x, where x is a specified speed in mph (e.g., x = 60), and d is the weight (distance in miles) of the edge. Note that x is assumed the same for all edges in Program 1.

c) An additional amount of time on an edge is a function of the preference value, vedge, assigned to it. In particular, the additional time spent on an edge is computed by add_time_on_edge (vedge), which for Program 1 will simply be a call to time_at_location (vedge).

d) time_estimate (roadtrip, x) returns the sum of times at all locations and on all edges of the road trip using x as the assumed travel speed on edges. The roadtrip argument can be any graph of valid locations and edges – it need not be round trip, and it need not be connected (again, because this function will be used to estimate times of partially constructed round-trips at intermediate points during search, as well as on final, fully-connected round trips).

5) Write a top-level function

RoundTripRoadTrip (startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile)

that does an anytime search for round-trip road trips.

Total round-trip road trip preference and time should not include preference and implied time of startLoc (which is also the end location of a round trip). Consideration of the startLoc can happen either inside or outside the total_preference and time_estimate functions — your choice.
Road trips that are returned cannot exceed maxTime (in hours) and the search should be designed such that round trips with higher total preference are more likely to be found earlier than later. That is, if time_estimate(RTRTj, x_mph) <= maxTime and time_estimate(RTRTk, x_mph) <= maxTime and total_preference (RTRTj, x_mph) > total_preference (RTRTk) then its desirable (though not required) that RTRTj be returned before RTRTk.
Note that there is no requirement that a round-trip road trip visit any location at most once, but there should be a bias for avoiding multiple visits to the same location (other than the start/end location). The leeway in making this a vias instead of an absolute requirement is that to reach some locations (e.g., a location on a “cul-de-sac”) may require multiple visits to other locations, on the way  to and from. In any case your time estimates and total preference (utility) should not double count visits to the same location on the same road trip.
A Test Run

A test run results from a single call to RoundTripRoadTrip (startLoc, LocFile, EdgeFile, maxTime, x_mph, resultFile).

Such a call will result in a sequence of solution paths (round trips), with the user answering ‘yes’ so that at least three solution paths result in the run.

Each solution path will be written to the screen and the resultFile, using the same format for both cases. Each solution path is identified by a unique program-generated solutionLabel. The user will then be asked whether another solution should be returned. If ‘yes’, then the anytime search continues. These steps can occur indefinitely until the user answers ‘no’, or until there are no more candidate solutions, at which time the search terminates. Again, there should be at least solution paths found per run.

Each solution path is essentially a sequence of edges that connect the start location back to the start location, with intermediate locations interLoci.

solutionLabel   startLoc  maxTime x_mph
1. startLoc   interLoc1 edgeLabel  edgePreference  edgeTime  interLoc1Pref  interLoc1Time
2. interLoc1 interLoc2 edgeLabel  edgePreference  edgeTime  interLoc2Pref  interLoc2Time
…
N-1. interLocN-1 startLoc edgeLabel  edgePreference  edgeTime
startLoc  TotalTripPreference  TotalTripDistance  TotalTripTime

Each solution path is followed by a new blank line.

Each line of a solution path, except the first and last, corresponds to an edge between two locations with those labels given first and second on the line, followed by the edgeLabel between them, the edge preference value, the edge time (base + additional), and the preference and time of the second of the locations on the line.

After the user answers ‘no’ in a test run, a summary of the entire run should be written to the screen and to the resultFile. The summary includes

the average instrumented runtime of all continuations of the search (that is, search time per solution paths, and do not include the time that search is paused between solutions so that you can answer yes/no).
The maximum TotalTripPreference found across all solution paths (of which  there should be at least three)
The average TotalTripPreference found across all solution paths (of which  there should be at least three)
The minimum TotalTripPreference found across all solution paths (of which  there should be at least three)
Again, this summary will be written to screen and to the resultFile immediately following the last solution returned for a given test run.

While the data files provides undirected edges in the road network, each line of a solution gives a directed edge, in which the direction is from first location to the second location in the line of output. So your program will have to revise (or reinterpret) the undirected edges in the data file to be directed edges as appropriate in each solution path. (You could also interpret this as the data files contains a bidirectional edge, but in your output, one direction is specified). This shouldn’t be difficult.

Part A) Deliverables

You are to do at least three test runs (of at least three solution paths each), where there will be a separate resultFile for each run.

You will be submitting one BIG file and at least three smaller ones

a fully documented program listing, to the extent possible placed in one file (for purposes of grading). Before the beginning of the program listing include (in order):
(a) comments at the top of file with Team number, team members;

(b) comments on how to run your code;

(c) anything notable about runtime or search strategy? (e.g., even though this is overtly a utility driven search in which goal-driven methods are not directly applicable, was there any way that you were able to use path cost and heuristic distance?)

Then give the program listing, with

(d) header comments for major functions.

After the end of the program listing give:

(e) qualitative comments on test runs (e.g., did any of your test cases correspond to situations in which no solution was possible? did your solutions monotonically decrease in value? did all test runs lead to solutions that met the hard time constraints?

(f) quantitative summary of test runs, notably

(i) the average of instrumented runtime of all continuations of all test runs (not including the time that search is paused between solutions so that you can answer yes/no).

(ii) the average  maximum TotalTripPreference found across all solution paths of all test runs;

(iii) the average TotalTripPreference found across all solution paths of all test runs;

(iv) the average minimum TotalTripPreference found across all solution paths of all test runs.

resultFiles for at least 3 test runs (i.e., 3 anytime searches of at least three continuances each).
 
 
