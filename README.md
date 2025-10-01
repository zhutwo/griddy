# griddy

A set of 3 grid search hyper-parameter tuning utilities I developed extra-curricularly while taking the CS7642 Deep Learning class at Georgia Tech.

### griddy_a1

Assignment 1 did not need grid search - you can get passing results by just manually tuning. The starter code was written in a way that did not interface easily with existing grid search tools, which is why I made this as a work around. In hindsight, that was probably a clue to the unnecessary path I was going down. However, I was fresh off the CS7641 Machine Learning grind (a research paper based class) and that primed me for overkill.

### griddy_a2

Assignment 2 featured a leaderboard competition where students were tasked with developing and training a pytorch model on the MNIST dataset. Models were scored on a metric weighing classification accuracy as well as model size. Since keeping model size down heavily influenced the overall score, I developed a strategy of implementing a CNN-ResNet model with an architecture that could be dynamically configured alongside the usual hyper-parameters using a modified version of `griddy_a1`. This ultimately led to a final rank of 5th amongst 646 entrants.

<img width="943" height="413" alt="griddy_leaderboard" src="https://github.com/user-attachments/assets/d5433eb7-c2a1-4318-9e30-4e780b667ca9" />

### griddy_tuna


`griddy_tuna` is a version made for the final group project as a means to wrap Optuna functionality in a format that was already familiar to my teammates, thereby sparing them from needing to RTFM.



