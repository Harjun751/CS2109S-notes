---
  title: Search | Week 2
  layout: default.md
  pageNav: 4
  pageNavTitle: "Topics"
---

# Search
A search problem refers to a type of problem where the goal is to find a state or a path to a state from a set of possible states by exploring various possibilities. It either returns a possible solution, or a failure indication. Often, we talk about the *optimal* solution - the path that is the shortest.

## Designing an agent to do search
<div id="formulation">
When designing a search agent, we should go through *problem formulation*.

We decide on:

1. State

2. Initial State

3. Goal State

4. Actions

5. Transition model

6. Action Cost
</div>

<panel header="Romania Problem Formulation">
Problem formulation
States - Cities
Initital State - Sebu
Goal - Bucharest
Actions - Go to adj city
Transition model - Current state = selected city
Action cost - edges
</panel>

## Search pseudocode
```
create frontier             # queue, pque, stack

insert node(initial state) to frontiner
while frontiner is not empty:
    node = frontier.pop()
    if node.state is goal: return solution

    for action in actions(node.state):
        next_state = transition(node.state, action);
        frontier.add(Node(next_state))
return failure
```


## Terminology
Problem (graph)
Search tree - the DS made by the search
Node (in search tree) ---contains--> States (in node)

## Evaluation criteria
Time complexity - Number of nodes generated/expanded
Space complexity - maximum number of nodes in memory

### Completeness
An algorithm is complete if for every problem instance, it will find a solution if one exists.

### Optimal
Algo is optimal if the solution produced is guaranteed to be optiomal

Optimal & Incomplete algorithms exist.

# Uninformed Search algorithms
Has no idea how "good" a state is - we don't know how close a state is to the goal.
Only know problem formulation and action cost.

## BFS
BFS - apply a queue to the search pseudocode and we got it!

Time Complexity - Exponentional
Space Complexity - Exponential
Completeness - If a problem has a finite branching factor, and there exists a goal at a finite depth, BFS is complete; it will terminate and return a solution
Optimality: Assume all actions have equal positive cost. If a goal exists at a finite depth, BFS returns a solution with minimal path cost.

## Uniform Cost Search (UCS)
UCS - apply a priority queue 

Time complexity - Exponential
Space - Exponential

Completeness - Finite branching factor, and every action cost must be positive and non zero. Then, if optimal path cost is < inf, then UCS is complete

Optimality - Positive edge cost implies optimality.


## DFS
DFS - Apply a stack 

Time complexity - Exponential
Space - Polynomial (only need to keep track of one path - the max depth)

Complete - Not complete. When DFS is infinite
Optimal - No, the optimal solution may be in a shallower depth.


## Search With visited memory
Add a visited DS - then check if node has been visited before traversing further.


# Informed Search Algorithms
"Search with extra info"

## Heuristic
Recall:
- Path cost is the cost of a path from any state to any state.
- Optimal path cost is the lowest-cost path

A Heuristic is an estimate of the optimal path cost from any state to the goal state.
All heuristic functions must be non-negative and have h(goal) = 0

## Best-first search 
Create a priority queue where f(n) = h(n) 

## A* Search
Create a priority queue where f(n) = g(n) + h(n) 
g(n) represents the cost to reach n, and h(n) represents the cost to reach the goal from n.

Time complexity: Exponential
Space: Exponential

Completeness: Similar to UCS, but additional conditions for heuristic
Optimality: Similar to UCS, but additional conditions for heuristic

### Admissable Heuristics
A heuristic h(N) is admissible if for every node n, 0 <= h(n) <= h*(n), where h*(n) is the optimal path cost to reach the goal state from n.
I.e., it never overestimates the actual true cost.

If admissible, A* search without visited memory is complete and optimal (with the UCS constraint)

#### Inventing Admissable Heuristic
A problem with fewer restrictions on the actions is called a relaxed problem.

The cost of an optimal solution to a relaxed problem is an admissible heuristic for the original problem.

#### Dominance
if h1(n) >= h2(n), then h1 dominates h2.
If admissible, then h1 is better for search (since it's a better estimate)

### Consistent Heuristics
A h(n) is consistent if for every node n, every successor n' of n, h(n) <= c(n,a,n') + h(n')
Triangle inequality

Theorem - if h(n) is consistent, A* search with visited memory is complete and optimal (with the UCS constraint)

Consistentcy => Admissability.

## Search Strategies
Without visited memory, search algorithms may not terminate.
- Depth-first search uses polynomial memory but may not terminate.
- Strategy: Set a strict resource limit. One example is max depth
- This gives birth to a search with polynomial memory, but is complete and optimal

### Depth-limited search
Limited depth to l
Backtrack when limit is hit.

Time complexity: Exponential
Space: Polynomial
Completeness: No
Optimal: No, if used with DFS.

### Iterative Deepening search

Time complexity: Exponential (with overhead)
Space: Polynomial
Complete: Yes, under the same conditions that guarantee BFS completeness
Optimal: Yes, under the same conditions that guarantee BFS optimality.
