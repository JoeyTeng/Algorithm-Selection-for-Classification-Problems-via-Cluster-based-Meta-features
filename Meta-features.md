| feature                                  | paper                         | frequency |
| ---------------------------------------- | ----------------------------- | --------- |
| Computational complexity                 | Giraud(1998)[^1]              | 0         |
| Expressiveness                           | ~                             | 0         |
| Compactness                              | ~                             | 0         |
| Comprehensibility                        | ~                             | 0         |
| Prior knowledge encoding                 | ~                             | 0         |
| The degree of correlation between features and the target concept | Aha(1992)[^2]                 | 0         |
| The distribution of examples within concepts disjuncts | ~                             | 0         |
| The distribution of examples among concepts | ~                             | 0         |
| Number of examples                       | Brazdil(1998)[^3]             | 0         |
| Number of attributes                     | ~, Brazdil et al. (2009)[^9]  | 0         |
| Standard deviation ratio for each feature | ~                             | 0         |
| Skewness                                 | ~                             | 0         |
| Kurtosis                                 | ~                             | 0         |
| noise-signal ratio                       | ~                             | 0         |
| Properties of a decision-tree: Nodes per feature | Bensusan et al.(2000)[^4]     | 0         |
| Maximum tree depth                       | ~                             | 0         |
| Shape                                    | ~                             | 0         |
| Tree Imbalance                           | ~                             | 0         |
| Accuracy                                 | Keller et al. (2000)[^5]      | 0         |
| Storage space                            | ~                             | 0         |
| Running time                             | ~                             | 0         |
| Number of instances                      | Asuncion and Newman(2007)[^6] | 0         |
| Number of classes                        | ~, Brazdil et al. (2009)[^9]  | 0         |
| Number of prototypes per class           | ~                             | 0         |
| Number of relevant and irrelevant attributes | ~                             | 0         |
| The distribution range of the instances and prototypes | ~                             | 0         |
| Difficulty of a classification problem   | Vilalta and Drissi(2002)[^7]  | 0         |
| Ratio of examples to features            | Brazdil et al. (2009)[^9]     | 0         |
| Degree of correlation between features and target concept | ~                             | 0         |
| Average class entropy                    | ~                             | 0         |
| Landmarking: Performance of simple algorithms | Pfahringer et al. (2000)[^10] | 0         |

[^1]: Giraud-Carrier Christophe (1998). Beyond Predictive Accuracy: What?. Proceedings of the ECML-98 Workshop on Upgrading Learning to Meta-Level: Model Selection and Data Transformation, 78–85. Technical University of Chemnitz.

[^2]: Aha David, W. (1992). Generalizing from Case Studies: A Case Study. Proceedings of the Ninth International Workshop on Machine Learning, 1–10. Morgan Kaufman.

[^3]: Brazdil Pavel, B. (1998). Data Transformation and model selection by experimentation and meta-learning. Proceedings of the ECML-98 Workshop on Upgrading Learning to Meta-Level: Model Selection and Data Transformation, 11–17. Technical University of Chemnitz.
[^4]: Bensusan Hilan & Giraud-Carrier Christophe (2000). Casa Batlo in Passeig or landmarking the expertise space. Eleventh European Conference on Machine Learning, Workshop on Meta-Learning: Building Automatic Advice Strategies for Model Selection and Method Combination. Barcelona, Spain. /// Bensusan Hilan, Giraud-Carrier Christophe & Kennedy, C. J. (2000). A High-Order Approach to Meta-Learning. Eleventh European Conference on Machine Learning, Workshop on Meta-Learning: Building Automatic Advice Strategies for Model Selection and Method Combination. Barcelona, Spain.
[^5]: Keller Jorg, Paterson Iain & Berrer Helmutt (2000). An Integrated Concept for Multi- Crieria-Ranking of Data-Mining Algorithms. Eleventh European Conference on Machine Learning, Workshop on Meta-Learning: Building Automatic Advice Strategies for Model Selection and Method Combination. Barcelona, Spain.
[^7]: Vilalta R, Drissi Y (2002b) A characterization of difficult problems in classification. In: Proceedings of the 6th European conference on principles and practice of knowledge discovery in databases, Helsinki, Finland
[^9]: Brazdil P, Giraud-Carrier C, Soares C, Vilalta R (2009) Metalearning: applications to data mining. Springer, Berlin

[^10]: Pfahringer B, Bensusan H, Giraud-Carrier C (2000) Meta-learning by landmarking various learning algorithms. In: Proceedings of the seventeenth international conference on machine learning. Morgan Kaufmann, pp 743–750

#### StatLog & METAL

| Measures           | Definitions                              |
| ------------------ | ---------------------------------------- |
| Simple             |                                          |
| N                  | Number of examples                       |
| p                  | Number of attributes                     |
| q                  | Number of classes                        |
| Bin.att            | Number of binary attributes              |
| Cost               | Cost matrix indicator                    |
| Statistical        |                                          |
| SD                 | Standard deviation ratio (geometric mean) |
| corr.abs           | Mean absolute correlation of attributes  |
| cancor1            | First canonical correlation              |
| fract1             | Fraction separability due to cancor1     |
| skewness           | Skewness—mean of $\|E (X − \mu)^3\|/\sigma^3$ |
| kurtosis           | Kurtosis—mean of $\|E (X − \mu)^4\|/\sigma^4$ |
| Information theory |                                          |
| $H(C)$             | Entropy (complexity) of class            |
| $\bar{H}  ( X )$   | Mean entropy (complexity) of attributes  |
| $\bar{M}( C , X )$ | Mean mutual information of class and attributes |
| EN.attr            | Equivalent number of attributes $H(C)/\bar{M}(C, X)$ |
| NS.ratio           | Noise-signal ratio $(\bar{H}(X){−}\bar{M}(C, X))/\bar{M}(C, X)$ |
