Program 3 requires that you expand upon the road trip recommender system in two ways.

Regression tree guidance. Modify the system so that it consults a regression tree to identify the preferences of a user. In principle this regression tree (or forest of such trees) would be learned from a user’s (training) data, but for purposes of illustrating this functionality you are to hand craft a regression tree. The intent is that in evaluating the overall utility of a road trip, the regression tree is consulted for each location and edge on the candidate road trip, where individual internal nodes of the regression tree are individual themes that are represented (or not) on edges and locations, a path of internal nodes represents a set of theme values (present or absent) found (or not) at a location or edge, and leaves are utility values (in the interval [0 to 1]) that are associated with the corresponding path’s set of theme values. Implicit in a regression tree are implementations of concepts such as substitutability, additive utility, complementary and substitute factors (12.1)
You must define a function to compute a road trip’s predicted overall utility from the assessments of each of its location/edge predicted utilities.
User control. Modify the system to include more user control into search. Notably, allow the user to provide a set of required locations and a set of forbidden locations that must be part of the road trip, or not part of the road trip. Returned road trips should still tend to be highest utility and to not violate time constraints, under the constraints provided.
Deliverable: Submit the following.

Fully documented source code listing, or pointer to one that is accessible to Doug and Kyle, with a ReadMe file.
A report of approximately 5 pages that includes
An opening paragraph overview of what you’ve done (1. Introduction)
Two hand-crafted regression trees that you considered, and the one you selected for tests. reported herein.
A couple of paragraphs on experimental design on tests with inclusion and exclusion constraints, as well as a description of your formula for computing overall utility (2. Experimental Design))
The experimental results showing “a table” (or subsections) of three road trip inclusion/exclusion constraints, three road trips found (using same format as Program 1), road trip utilities, and average runtime over the three road trips found (as in Program 1) for each set of constraints (3. Experimental Results)
If you used AI to implement, evaluate, document the systems then 1 or more paragraphs of the experience and any insights on  strengths and weaknesses (4. General Discussion)
5. Concluding remarks
If this comes out to fewer than 5 pages, or more than 5 pages, 1.5 spacing, 1 inch margins, 12 pt  font, then that’s fine.

See the course schedule for the due date of the deliverable.

Looking ahead: In Program 4 you will interface Program 3 with an LLM API, which will be used to summarize a road trip in narrative (and hopefully enticing) form.
