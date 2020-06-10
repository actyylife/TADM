# TADM
This repository is a python implementation of "A Topology-Adaptive Deep Model for Power System Multifault Diagnosis".

# Abstract
Quickly identifying faulty sections is tremendously important for power systems, yet challenging due to handling the variations of complex alarm patterns. Existing works have focused on finding fault section clues solely from alarm information (and ignoring power system topology information). So they are only sensitive to alarms from power systems with pre-assumed topology structures, and encounter difficulties when a system's topology changes. To adapt to unknown or varying system topologies, here we present a Topology-Adaptive Deep Model (TADM) for power system multifault diagnosis. TADM mines the underlying mapping from alarm and topology information to each section's fault status. It consists of a deep iterative network (DIN), a one-layer fully connected network (FCN), and section-wise multifault diagnosis (SWMD) subnetwork. TADM first models a fault power system as a graph, from which DIN iteratively integrates the alarm and topology information in the region from each node to its T-hop neighbors, and learns their local correlation. Limited to T's size, FCN then combines all local correlations to determine the global correlation between alarm and topology information across the entire power system. To implement multifault diagnosis, learned local and global correlations serve as topology-related fault representations for input as an SWMD (to predict all sections' fault states one by one). A comprehensive experimental study demonstrates that TADM outperforms state-of-the-art models in both multifault diagnosis and adapting to system topologies.