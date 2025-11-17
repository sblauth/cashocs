# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/sblauth/cashocs/blob/coverage/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| cashocs/\_\_init\_\_.py                                                            |       40 |        0 |    100% |           |
| cashocs/\_cli/\_\_init\_\_.py                                                      |        3 |        0 |    100% |           |
| cashocs/\_cli/\_convert.py                                                         |       22 |        1 |     95% |        93 |
| cashocs/\_cli/\_extract\_mesh.py                                                   |       22 |       16 |     27% |31-70, 80-89, 99 |
| cashocs/\_constraints/\_\_init\_\_.py                                              |        7 |        0 |    100% |           |
| cashocs/\_constraints/constrained\_problems.py                                     |      139 |        8 |     94% |146, 308, 323, 343-344, 546, 709, 774 |
| cashocs/\_constraints/constraints.py                                               |       81 |        0 |    100% |           |
| cashocs/\_constraints/deflated\_problems.py                                        |      202 |       46 |     77% |145, 210-221, 270, 285, 305-306, 323, 463, 753, 861-907, 923-935, 945-952, 982-1032 |
| cashocs/\_constraints/solvers.py                                                   |      185 |        7 |     96% |83-86, 426-427, 508-509 |
| cashocs/\_database/\_\_init\_\_.py                                                 |        0 |        0 |    100% |           |
| cashocs/\_database/database.py                                                     |       16 |        0 |    100% |           |
| cashocs/\_database/form\_database.py                                               |       10 |        0 |    100% |           |
| cashocs/\_database/function\_database.py                                           |       21 |        0 |    100% |           |
| cashocs/\_database/geometry\_database.py                                           |       23 |        2 |     91% |     69-70 |
| cashocs/\_database/parameter\_database.py                                          |       30 |        0 |    100% |           |
| cashocs/\_exceptions.py                                                            |       54 |        4 |     93% |150-151, 193-194 |
| cashocs/\_forms/\_\_init\_\_.py                                                    |        8 |        0 |    100% |           |
| cashocs/\_forms/control\_form\_handler.py                                          |      142 |       11 |     92% |30, 112-130, 146-147, 203 |
| cashocs/\_forms/form\_handler.py                                                   |       28 |        8 |     71% |     55-63 |
| cashocs/\_forms/general\_form\_handler.py                                          |       82 |       10 |     88% |49-55, 193, 198, 209-210 |
| cashocs/\_forms/shape\_form\_handler.py                                            |      270 |       24 |     91% |169, 207-208, 390, 412-418, 453, 570, 695-701, 711-724, 845-848 |
| cashocs/\_forms/shape\_regularization.py                                           |      220 |       10 |     95% |202, 211, 297, 305, 399, 474, 482, 503, 631, 639 |
| cashocs/\_optimization/\_\_init\_\_.py                                             |        0 |        0 |    100% |           |
| cashocs/\_optimization/cost\_functional.py                                         |      160 |        7 |     96% |234, 247, 272, 432, 465-466, 521 |
| cashocs/\_optimization/line\_search/\_\_init\_\_.py                                |        4 |        0 |    100% |           |
| cashocs/\_optimization/line\_search/armijo\_line\_search.py                        |       72 |        6 |     92% |216, 231-236 |
| cashocs/\_optimization/line\_search/line\_search.py                                |       71 |        2 |     97% |   68, 141 |
| cashocs/\_optimization/line\_search/polynomial\_line\_search.py                    |       96 |       18 |     81% |82, 85-87, 93-95, 137, 140, 176-186, 211, 246, 248, 343-346 |
| cashocs/\_optimization/optimal\_control/\_\_init\_\_.py                            |        3 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/box\_constraints.py                        |      112 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/control\_variable\_abstractions.py         |       64 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/optimal\_control\_problem.py               |      116 |        2 |     98% |  346, 454 |
| cashocs/\_optimization/optimization\_algorithms/\_\_init\_\_.py                    |        6 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/callback.py                        |       24 |        4 |     83% |51-52, 67-68 |
| cashocs/\_optimization/optimization\_algorithms/gradient\_descent.py               |       20 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/l\_bfgs.py                         |      166 |        1 |     99% |       170 |
| cashocs/\_optimization/optimization\_algorithms/ncg.py                             |       94 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/newton.py                          |       33 |        7 |     79% | 77-85, 91 |
| cashocs/\_optimization/optimization\_algorithms/optimization\_algorithm.py         |      196 |       14 |     93% |234, 276-279, 295-296, 344-346, 376-387 |
| cashocs/\_optimization/optimization\_problem.py                                    |      190 |       12 |     94% |214, 249, 425, 487-490, 541-543, 700-701, 792-793 |
| cashocs/\_optimization/optimization\_variable\_abstractions.py                     |       22 |        2 |     91% |   180-181 |
| cashocs/\_optimization/shape\_optimization/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| cashocs/\_optimization/shape\_optimization/shape\_optimization\_problem.py         |      143 |       17 |     88% |55, 277-285, 342-344, 387-388, 408, 479-482, 487-496 |
| cashocs/\_optimization/shape\_optimization/shape\_variable\_abstractions.py        |       46 |        4 |     91% |127-129, 166 |
| cashocs/\_optimization/topology\_optimization/\_\_init\_\_.py                      |        2 |        0 |    100% |           |
| cashocs/\_optimization/topology\_optimization/bisection.py                         |       69 |        0 |    100% |           |
| cashocs/\_optimization/topology\_optimization/descent\_topology\_algorithm.py      |       69 |        2 |     97% |  135, 162 |
| cashocs/\_optimization/topology\_optimization/topology\_optimization\_algorithm.py |      202 |       20 |     90% |31, 163, 187-194, 258, 271, 312-313, 319-325, 330-332, 359-363, 474, 506, 542 |
| cashocs/\_optimization/topology\_optimization/topology\_optimization\_problem.py   |       91 |        7 |     92% |281, 285, 331-332, 358-362 |
| cashocs/\_optimization/topology\_optimization/topology\_variable\_abstractions.py  |       35 |       13 |     63% |76, 80-83, 87-90, 120-127, 136, 151, 160, 174 |
| cashocs/\_pde\_problems/\_\_init\_\_.py                                            |        7 |        0 |    100% |           |
| cashocs/\_pde\_problems/adjoint\_problem.py                                        |       56 |        1 |     98% |       110 |
| cashocs/\_pde\_problems/control\_gradient\_problem.py                              |       41 |        4 |     90% | 77, 83-85 |
| cashocs/\_pde\_problems/hessian\_problems.py                                       |      174 |        0 |    100% |           |
| cashocs/\_pde\_problems/pde\_problem.py                                            |       13 |        1 |     92% |        56 |
| cashocs/\_pde\_problems/shape\_gradient\_problem.py                                |      136 |        8 |     94% |86, 195-199, 237, 363, 366-368 |
| cashocs/\_pde\_problems/state\_problem.py                                          |       88 |        8 |     91% |77, 82, 124, 161, 268-274 |
| cashocs/\_typing.py                                                                |       23 |       23 |      0% |     20-67 |
| cashocs/\_utils/\_\_init\_\_.py                                                    |       27 |        0 |    100% |           |
| cashocs/\_utils/forms.py                                                           |       62 |        0 |    100% |           |
| cashocs/\_utils/helpers.py                                                         |       72 |        1 |     99% |        70 |
| cashocs/\_utils/interpolations.py                                                  |       44 |       18 |     59% |27, 275-449 |
| cashocs/\_utils/linalg.py                                                          |      185 |       21 |     89% |78, 140, 158, 331-332, 389-392, 507-508, 638-642, 662-674 |
| cashocs/geometry/\_\_init\_\_.py                                                   |       18 |        0 |    100% |           |
| cashocs/geometry/boundary\_distance.py                                             |       63 |        1 |     98% |       154 |
| cashocs/geometry/deformations.py                                                   |       64 |        4 |     94% |130, 139, 144, 203 |
| cashocs/geometry/measure.py                                                        |       39 |        1 |     97% |       202 |
| cashocs/geometry/mesh.py                                                           |      149 |        2 |     99% |    80, 84 |
| cashocs/geometry/mesh\_handler.py                                                  |      265 |       26 |     90% |66, 214, 225, 229, 408, 410, 429, 438, 446, 454, 462, 470, 478, 493-494, 559, 652-672 |
| cashocs/geometry/mesh\_testing.py                                                  |       71 |        4 |     94% |179, 183-187, 245 |
| cashocs/geometry/quality.py                                                        |      104 |        7 |     93% |325, 363, 398, 459, 489, 513, 542 |
| cashocs/io/\_\_init\_\_.py                                                         |       17 |        0 |    100% |           |
| cashocs/io/config.py                                                               |      137 |        4 |     97% |32, 71, 738-739 |
| cashocs/io/function.py                                                             |       32 |       10 |     69% |     62-77 |
| cashocs/io/managers.py                                                             |      252 |       17 |     93% |366, 381, 403, 633-639, 642-649, 674-678 |
| cashocs/io/mesh.py                                                                 |      338 |       39 |     88% |93, 148-152, 300-324, 372-373, 405, 564-565, 569-571, 643, 674, 741, 745-746, 786, 797-803, 835 |
| cashocs/io/output.py                                                               |       57 |        0 |    100% |           |
| cashocs/log.py                                                                     |      142 |       15 |     89% |95, 117, 166, 286, 307-314, 323, 327, 331 |
| cashocs/mpi.py                                                                     |        2 |        0 |    100% |           |
| cashocs/nonlinear\_solvers/\_\_init\_\_.py                                         |       11 |        0 |    100% |           |
| cashocs/nonlinear\_solvers/linear\_solver.py                                       |       35 |        7 |     80% |     88-96 |
| cashocs/nonlinear\_solvers/newton\_solver.py                                       |      194 |       26 |     87% |122-127, 167-170, 189-192, 230-232, 259, 264, 317-319, 358, 366-372, 387-388, 400, 402, 435-436 |
| cashocs/nonlinear\_solvers/picard\_solver.py                                       |       82 |        7 |     91% |54-57, 164-166, 174 |
| cashocs/nonlinear\_solvers/snes.py                                                 |      106 |       10 |     91% |114-119, 144-148, 216-218, 260 |
| cashocs/nonlinear\_solvers/ts.py                                                   |      215 |       38 |     82% |132-135, 140-145, 175-179, 188-191, 197, 229, 231, 245, 279-280, 315-321, 374-376, 424, 434-438, 447, 450, 453, 512, 532, 534-535 |
| cashocs/space\_mapping/\_\_init\_\_.py                                             |        3 |        0 |    100% |           |
| cashocs/space\_mapping/optimal\_control.py                                         |      378 |       40 |     89% |172, 273, 356-361, 461, 548-549, 677-679, 686-688, 713-745, 930, 959-961, 971-973, 987-988 |
| cashocs/space\_mapping/shape\_optimization.py                                      |      392 |       53 |     86% |174, 276, 351-352, 446, 523-524, 539-566, 687-688, 705-707, 713-715, 745-783, 955, 982-984, 994-996, 1010-1011 |
| cashocs/verification.py                                                            |      131 |        3 |     98% |234-235, 267 |
|                                                                          **TOTAL** | **7939** |  **684** | **91%** |           |


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