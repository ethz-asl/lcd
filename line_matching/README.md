The following package is used to match lines from one frame to another, based on the distance between their descriptors/embeddings. *NOTE: in the current implementation, the embeddings retrieved from the neural networks are not meant to be line-feature descriptors, but are rather used to form clusters associated to instances (cf. main README). Therefore, one should not expect line-to-line matching, but rather [line from one instance]-to-[line of the same instance] matching*.  
This package is ROS-independent.

### Libraries
- **line_matching**: Library to match the lines.

  _Classes_:
  - `MatchRatingComputer`: Abstract class that can be used to implement 'distances' (e.g., Manhattan distance, Euclidean distance) by means of which the descriptors/embeddings of the lines can be compared for matching;
  - `LineMatcher`: Main class. For each frame it stores the lines detected (with their descriptors/embeddings) and the original image from which they were extracted. Then, it matches the lines from one frame to those from another frame and it displays matches;
  - `FixedSizePriorityQueue`: Auxiliary class that implements a fixed-size priority queue. Used to store only the `n` best matches for each line, rather than all the matches.
