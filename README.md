# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/sblauth/cashocs/blob/coverage/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| cashocs/\_\_init\_\_.py                                                            |       37 |        0 |    100% |           |
| cashocs/\_cli/\_\_init\_\_.py                                                      |        3 |        0 |    100% |           |
| cashocs/\_cli/\_convert.py                                                         |      123 |       10 |     92% |119, 123-124, 164, 175-180, 210, 295 |
| cashocs/\_cli/\_extract\_mesh.py                                                   |       22 |       16 |     27% |30-68, 78-87, 97 |
| cashocs/\_constraints/\_\_init\_\_.py                                              |        5 |        0 |    100% |           |
| cashocs/\_constraints/constrained\_problems.py                                     |      139 |        8 |     94% |147, 309, 324, 344-345, 547, 710, 775 |
| cashocs/\_constraints/constraints.py                                               |       81 |        0 |    100% |           |
| cashocs/\_constraints/solvers.py                                                   |      186 |        7 |     96% |83-86, 429-430, 511-512 |
| cashocs/\_database/\_\_init\_\_.py                                                 |        0 |        0 |    100% |           |
| cashocs/\_database/database.py                                                     |       16 |        0 |    100% |           |
| cashocs/\_database/form\_database.py                                               |       10 |        0 |    100% |           |
| cashocs/\_database/function\_database.py                                           |       21 |        0 |    100% |           |
| cashocs/\_database/geometry\_database.py                                           |       24 |        2 |     92% |     69-70 |
| cashocs/\_database/parameter\_database.py                                          |       30 |        0 |    100% |           |
| cashocs/\_exceptions.py                                                            |       54 |        4 |     93% |150-151, 193-194 |
| cashocs/\_forms/\_\_init\_\_.py                                                    |        8 |        0 |    100% |           |
| cashocs/\_forms/control\_form\_handler.py                                          |      142 |       11 |     92% |30, 112-130, 146-147, 203 |
| cashocs/\_forms/form\_handler.py                                                   |       28 |        8 |     71% |     55-63 |
| cashocs/\_forms/general\_form\_handler.py                                          |       82 |       10 |     88% |49-55, 193, 198, 209-210 |
| cashocs/\_forms/shape\_form\_handler.py                                            |      270 |       23 |     91% |167, 205, 384, 406-412, 447, 564, 690-696, 706-719, 840-843 |
| cashocs/\_forms/shape\_regularization.py                                           |      220 |       10 |     95% |202, 211, 297, 305, 399, 474, 482, 503, 631, 639 |
| cashocs/\_optimization/\_\_init\_\_.py                                             |        0 |        0 |    100% |           |
| cashocs/\_optimization/cost\_functional.py                                         |      129 |        4 |     97% |234, 247, 272, 432 |
| cashocs/\_optimization/line\_search/\_\_init\_\_.py                                |        4 |        0 |    100% |           |
| cashocs/\_optimization/line\_search/armijo\_line\_search.py                        |       73 |        6 |     92% |210, 225-230 |
| cashocs/\_optimization/line\_search/line\_search.py                                |       67 |        2 |     97% |   66, 139 |
| cashocs/\_optimization/line\_search/polynomial\_line\_search.py                    |       97 |       18 |     81% |83, 86-88, 94-96, 138, 141, 174-184, 207, 242, 244, 339-342 |
| cashocs/\_optimization/optimal\_control/\_\_init\_\_.py                            |        3 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/box\_constraints.py                        |      112 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/control\_variable\_abstractions.py         |       64 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/optimal\_control\_problem.py               |      115 |        2 |     98% |  345, 453 |
| cashocs/\_optimization/optimization\_algorithms/\_\_init\_\_.py                    |        6 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/callback.py                        |       23 |        4 |     83% |50-51, 66-67 |
| cashocs/\_optimization/optimization\_algorithms/gradient\_descent.py               |       20 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/l\_bfgs.py                         |      166 |        1 |     99% |       170 |
| cashocs/\_optimization/optimization\_algorithms/ncg.py                             |       94 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/newton.py                          |       33 |        7 |     79% | 77-85, 91 |
| cashocs/\_optimization/optimization\_algorithms/optimization\_algorithm.py         |      196 |       14 |     93% |234, 276-279, 295-296, 344-346, 376-387 |
| cashocs/\_optimization/optimization\_problem.py                                    |      188 |       12 |     94% |212, 247, 423, 485-488, 539-541, 698-699, 790-791 |
| cashocs/\_optimization/optimization\_variable\_abstractions.py                     |       22 |        2 |     91% |   180-181 |
| cashocs/\_optimization/shape\_optimization/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| cashocs/\_optimization/shape\_optimization/shape\_optimization\_problem.py         |      142 |       17 |     88% |54, 275-283, 340-342, 385-386, 406, 477-480, 485-493 |
| cashocs/\_optimization/shape\_optimization/shape\_variable\_abstractions.py        |       44 |        4 |     91% |125-127, 164 |
| cashocs/\_optimization/topology\_optimization/\_\_init\_\_.py                      |        2 |        0 |    100% |           |
| cashocs/\_optimization/topology\_optimization/bisection.py                         |       62 |        0 |    100% |           |
| cashocs/\_optimization/topology\_optimization/descent\_topology\_algorithm.py      |       69 |        2 |     97% |  135, 162 |
| cashocs/\_optimization/topology\_optimization/topology\_optimization\_algorithm.py |      195 |       24 |     88% |30, 154, 178-185, 249, 262, 303-304, 310-316, 321-323, 350-354, 407, 461-465, 497, 533 |
| cashocs/\_optimization/topology\_optimization/topology\_optimization\_problem.py   |       87 |        7 |     92% |277, 281, 327-328, 354-358 |
| cashocs/\_optimization/topology\_optimization/topology\_variable\_abstractions.py  |       35 |       13 |     63% |76, 80-83, 87-90, 120-127, 136, 151, 160, 174 |
| cashocs/\_pde\_problems/\_\_init\_\_.py                                            |        7 |        0 |    100% |           |
| cashocs/\_pde\_problems/adjoint\_problem.py                                        |       56 |        1 |     98% |       111 |
| cashocs/\_pde\_problems/control\_gradient\_problem.py                              |       41 |        4 |     90% | 77, 83-85 |
| cashocs/\_pde\_problems/hessian\_problems.py                                       |      174 |        0 |    100% |           |
| cashocs/\_pde\_problems/pde\_problem.py                                            |       13 |        1 |     92% |        58 |
| cashocs/\_pde\_problems/shape\_gradient\_problem.py                                |      136 |        8 |     94% |86, 195-199, 238, 362, 365-367 |
| cashocs/\_pde\_problems/state\_problem.py                                          |       88 |        8 |     91% |77, 82, 121, 158, 266-272 |
| cashocs/\_typing.py                                                                |       23 |       23 |      0% |     20-67 |
| cashocs/\_utils/\_\_init\_\_.py                                                    |       27 |        0 |    100% |           |
| cashocs/\_utils/forms.py                                                           |       62 |        0 |    100% |           |
| cashocs/\_utils/helpers.py                                                         |       71 |        1 |     99% |        69 |
| cashocs/\_utils/interpolations.py                                                  |       44 |       18 |     59% |27, 275-449 |
| cashocs/\_utils/linalg.py                                                          |      192 |       19 |     90% |77, 152, 347-348, 408-411, 534, 671-675, 695-707 |
| cashocs/geometry/\_\_init\_\_.py                                                   |       18 |        0 |    100% |           |
| cashocs/geometry/boundary\_distance.py                                             |       63 |        1 |     98% |       148 |
| cashocs/geometry/deformations.py                                                   |       66 |        5 |     92% |131-135, 143, 148, 207 |
| cashocs/geometry/measure.py                                                        |       40 |        1 |     98% |       203 |
| cashocs/geometry/mesh.py                                                           |      139 |        0 |    100% |           |
| cashocs/geometry/mesh\_handler.py                                                  |      262 |       26 |     90% |99, 241, 252, 256, 432, 434, 459, 469, 480, 491, 502, 513, 524, 539-540, 605, 696-716 |
| cashocs/geometry/mesh\_testing.py                                                  |       69 |        4 |     94% |175, 179-180, 237 |
| cashocs/geometry/quality.py                                                        |       93 |        6 |     94% |308, 346, 381, 444, 474, 498 |
| cashocs/io/\_\_init\_\_.py                                                         |       17 |        0 |    100% |           |
| cashocs/io/config.py                                                               |      138 |        4 |     97% |33, 72, 727-728 |
| cashocs/io/function.py                                                             |       21 |       15 |     29% |58-73, 93-99 |
| cashocs/io/managers.py                                                             |      253 |       17 |     93% |365, 379, 401, 632-638, 641-648, 674-678 |
| cashocs/io/mesh.py                                                                 |      241 |       32 |     87% |91, 146-150, 297-322, 348, 377-378, 411, 567-568, 572-574, 645, 678 |
| cashocs/io/output.py                                                               |       57 |        0 |    100% |           |
| cashocs/log.py                                                                     |      115 |       14 |     88% |114, 231-232, 255-262, 271, 275, 279 |
| cashocs/nonlinear\_solvers/\_\_init\_\_.py                                         |       11 |        0 |    100% |           |
| cashocs/nonlinear\_solvers/linear\_solver.py                                       |       35 |        7 |     80% |     88-96 |
| cashocs/nonlinear\_solvers/newton\_solver.py                                       |      196 |       26 |     87% |123-128, 166-169, 188-191, 229-231, 260, 265, 316-318, 357, 365-371, 386-387, 400, 402, 435-436 |
| cashocs/nonlinear\_solvers/picard\_solver.py                                       |       82 |        7 |     91% |54-57, 164-166, 174 |
| cashocs/nonlinear\_solvers/snes.py                                                 |      109 |       10 |     91% |114-119, 144-148, 217-219, 258 |
| cashocs/nonlinear\_solvers/ts.py                                                   |      197 |       28 |     86% |132-135, 140-145, 175-179, 188-191, 197, 226, 228, 241, 275-276, 312-318, 375-377, 433, 486, 501 |
| cashocs/space\_mapping/\_\_init\_\_.py                                             |        3 |        0 |    100% |           |
| cashocs/space\_mapping/optimal\_control.py                                         |      381 |       42 |     89% |165, 266, 349-354, 454, 541-542, 672-674, 681-683, 710-744, 929, 958-960, 970-972, 986-987 |
| cashocs/space\_mapping/shape\_optimization.py                                      |      395 |       55 |     86% |167, 269, 344-345, 439, 516-517, 532-559, 682-683, 700-702, 708-710, 742-782, 954, 981-983, 993-995, 1009-1010 |
| cashocs/verification.py                                                            |      123 |        3 |     98% |214-215, 247 |
|                                                                          **TOTAL** | **7610** |  **634** | **92%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/sblauth/cashocs/coverage/badge.svg)](https://htmlpreview.github.io/?https://github.com/sblauth/cashocs/blob/coverage/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/sblauth/cashocs/coverage/endpoint.json)](https://htmlpreview.github.io/?https://github.com/sblauth/cashocs/blob/coverage/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fsblauth%2Fcashocs%2Fcoverage%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/sblauth/cashocs/blob/coverage/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.