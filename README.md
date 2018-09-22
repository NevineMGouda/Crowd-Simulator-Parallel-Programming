# Crowd_Simulator_Parallel_Programming

Crowd (or Pedestrian) simulation is the study of the movement of people through various environments in order to understand the collective behaviour of crowds in different situations. Accurate simulations could be useful when designing airports or other public spaces, for understanding crowd behaviour departing sporting events or festivals, and for dealing with emergencies related to such situations. Understanding crowd behaviour in normal and abnormal situations helps improve the design of the public spaces for both efficiency of movement and safety, answering questions such as where the (emergency) exits should be put. This system simulates an area of people walking around. Each person, or agent, is moving towards a circular sequence of waypoints, in the 2D area. The starting locations and waypoints are specified in a scenario configuration file, which is loaded at the beginning of the program run. Agents aren't allowed to walk over each other, thus collision detection is implemented. 
With each update of the world two things happen:
1. The model is updated, allowing agents to act.
2. The graphical visualization is updated.

The time that passes for each such update thus decides the frame rate.

The project is done using the following tools and paradigms:
1. C++
2. CUDA
3. SIMD vector operations
4. c++ Threads
5. OpenMP
6. PThreads
7. Task parallelism
8. Race-free inter-thread interaction
9. Load balancing
10. Heterogeneous computation
