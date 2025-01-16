# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/sblauth/cashocs/blob/coverage/htmlcov/index.html)

| Name                                                                               |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| cashocs/\_\_init\_\_.py                                                            |       37 |        0 |    100% |           |
| cashocs/\_cli/\_\_init\_\_.py                                                      |        3 |        0 |    100% |           |
| cashocs/\_cli/\_convert.py                                                         |      134 |       12 |     91% |142, 153-158, 193, 204-209, 239, 325 |
| cashocs/\_cli/\_extract\_mesh.py                                                   |       22 |       16 |     27% |30-68, 78-87, 97 |
| cashocs/\_constraints/\_\_init\_\_.py                                              |        5 |        0 |    100% |           |
| cashocs/\_constraints/constrained\_problems.py                                     |      138 |        8 |     94% |143, 304, 319, 339-340, 536, 693, 757 |
| cashocs/\_constraints/constraints.py                                               |       81 |        0 |    100% |           |
| cashocs/\_constraints/solvers.py                                                   |      182 |        7 |     96% |78-81, 424-425, 506-507 |
| cashocs/\_database/\_\_init\_\_.py                                                 |        0 |        0 |    100% |           |
| cashocs/\_database/database.py                                                     |       16 |        0 |    100% |           |
| cashocs/\_database/form\_database.py                                               |       10 |        0 |    100% |           |
| cashocs/\_database/function\_database.py                                           |       21 |        0 |    100% |           |
| cashocs/\_database/geometry\_database.py                                           |       20 |        2 |     90% |     64-65 |
| cashocs/\_database/parameter\_database.py                                          |       30 |        0 |    100% |           |
| cashocs/\_exceptions.py                                                            |       54 |        4 |     93% |150-151, 193-194 |
| cashocs/\_forms/\_\_init\_\_.py                                                    |        8 |        0 |    100% |           |
| cashocs/\_forms/control\_form\_handler.py                                          |      142 |       11 |     92% |30, 112-130, 146-147, 203 |
| cashocs/\_forms/form\_handler.py                                                   |       24 |        8 |     67% |     50-58 |
| cashocs/\_forms/general\_form\_handler.py                                          |       82 |       10 |     88% |49-55, 193, 198, 209-210 |
| cashocs/\_forms/shape\_form\_handler.py                                            |      239 |       14 |     94% |165, 203, 359, 381-386, 421, 538, 646-657 |
| cashocs/\_forms/shape\_regularization.py                                           |      220 |       10 |     95% |202, 211, 297, 305, 401, 476, 484, 505, 633, 641 |
| cashocs/\_optimization/\_\_init\_\_.py                                             |        0 |        0 |    100% |           |
| cashocs/\_optimization/cost\_functional.py                                         |      129 |        4 |     97% |234, 247, 272, 432 |
| cashocs/\_optimization/line\_search/\_\_init\_\_.py                                |        4 |        0 |    100% |           |
| cashocs/\_optimization/line\_search/armijo\_line\_search.py                        |       72 |        6 |     92% |212, 227-232 |
| cashocs/\_optimization/line\_search/line\_search.py                                |       67 |        2 |     97% |   66, 139 |
| cashocs/\_optimization/line\_search/polynomial\_line\_search.py                    |       96 |       18 |     81% |83, 86-88, 94-96, 138, 141, 176-186, 209, 244, 246, 341-344 |
| cashocs/\_optimization/optimal\_control/\_\_init\_\_.py                            |        3 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/box\_constraints.py                        |      112 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/control\_variable\_abstractions.py         |       64 |        0 |    100% |           |
| cashocs/\_optimization/optimal\_control/optimal\_control\_problem.py               |      115 |        2 |     98% |  337, 445 |
| cashocs/\_optimization/optimization\_algorithms/\_\_init\_\_.py                    |        6 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/callback.py                        |       23 |        4 |     83% |50-51, 66-67 |
| cashocs/\_optimization/optimization\_algorithms/gradient\_descent.py               |       20 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/l\_bfgs.py                         |      166 |        1 |     99% |       170 |
| cashocs/\_optimization/optimization\_algorithms/ncg.py                             |       94 |        0 |    100% |           |
| cashocs/\_optimization/optimization\_algorithms/newton.py                          |       33 |        7 |     79% | 77-85, 91 |
| cashocs/\_optimization/optimization\_algorithms/optimization\_algorithm.py         |      196 |       14 |     93% |234, 276-279, 295-296, 344-346, 376-387 |
| cashocs/\_optimization/optimization\_problem.py                                    |      169 |        8 |     95% |392, 466-468, 625-626, 717-718 |
| cashocs/\_optimization/optimization\_variable\_abstractions.py                     |       22 |        2 |     91% |   180-181 |
| cashocs/\_optimization/shape\_optimization/\_\_init\_\_.py                         |        3 |        0 |    100% |           |
| cashocs/\_optimization/shape\_optimization/shape\_optimization\_problem.py         |      142 |       17 |     88% |54, 270-278, 334-336, 379-380, 400, 471-474, 479-487 |
| cashocs/\_optimization/shape\_optimization/shape\_variable\_abstractions.py        |       44 |        4 |     91% |125-127, 164 |
| cashocs/\_optimization/topology\_optimization/\_\_init\_\_.py                      |        2 |        0 |    100% |           |
| cashocs/\_optimization/topology\_optimization/bisection.py                         |       58 |        0 |    100% |           |
| cashocs/\_optimization/topology\_optimization/descent\_topology\_algorithm.py      |       69 |        2 |     97% |  135, 162 |
| cashocs/\_optimization/topology\_optimization/topology\_optimization\_algorithm.py |      193 |       23 |     88% |152, 176-183, 247, 260, 301-302, 308-314, 319-321, 348-352, 405, 459-463, 495, 531 |
| cashocs/\_optimization/topology\_optimization/topology\_optimization\_problem.py   |       87 |        7 |     92% |271, 275, 321-322, 348-352 |
| cashocs/\_optimization/topology\_optimization/topology\_variable\_abstractions.py  |       35 |       13 |     63% |76, 80-83, 87-90, 120-127, 136, 151, 160, 174 |
| cashocs/\_pde\_problems/\_\_init\_\_.py                                            |        7 |        0 |    100% |           |
| cashocs/\_pde\_problems/adjoint\_problem.py                                        |       46 |        1 |     98% |        99 |
| cashocs/\_pde\_problems/control\_gradient\_problem.py                              |       41 |        4 |     90% | 77, 83-85 |
| cashocs/\_pde\_problems/hessian\_problems.py                                       |      174 |        0 |    100% |           |
| cashocs/\_pde\_problems/pde\_problem.py                                            |       13 |        1 |     92% |        58 |
| cashocs/\_pde\_problems/shape\_gradient\_problem.py                                |       90 |        5 |     94% |84, 230, 233-235 |
| cashocs/\_pde\_problems/state\_problem.py                                          |       80 |        7 |     91% |73, 112, 149, 170, 240-244 |
| cashocs/\_typing.py                                                                |       19 |       19 |      0% |     20-62 |
| cashocs/\_utils/\_\_init\_\_.py                                                    |       25 |        0 |    100% |           |
| cashocs/\_utils/forms.py                                                           |       65 |        0 |    100% |           |
| cashocs/\_utils/helpers.py                                                         |       63 |        6 |     90% |69, 225-230 |
| cashocs/\_utils/interpolations.py                                                  |       41 |       17 |     59% |   271-444 |
| cashocs/\_utils/linalg.py                                                          |      179 |       20 |     89% |77, 152, 298, 347-348, 408-411, 534, 671-675, 695-707 |
| cashocs/geometry/\_\_init\_\_.py                                                   |       18 |        0 |    100% |           |
| cashocs/geometry/boundary\_distance.py                                             |       63 |        1 |     98% |       148 |
| cashocs/geometry/deformations.py                                                   |       66 |        5 |     92% |131-135, 143, 148, 207 |
| cashocs/geometry/measure.py                                                        |       40 |        1 |     98% |       203 |
| cashocs/geometry/mesh.py                                                           |      139 |        0 |    100% |           |
| cashocs/geometry/mesh\_handler.py                                                  |      256 |       26 |     90% |94, 236, 247, 251, 422, 424, 449, 459, 470, 481, 492, 503, 514, 529-530, 595, 686-706 |
| cashocs/geometry/mesh\_testing.py                                                  |       59 |        4 |     93% |158, 164-165, 217 |
| cashocs/geometry/quality.py                                                        |       93 |        6 |     94% |308, 346, 381, 444, 474, 498 |
| cashocs/io/\_\_init\_\_.py                                                         |       17 |        0 |    100% |           |
| cashocs/io/config.py                                                               |      139 |        3 |     98% |33, 711-712 |
| cashocs/io/function.py                                                             |       21 |       15 |     29% |58-73, 93-99 |
| cashocs/io/managers.py                                                             |      247 |       12 |     95% |365, 379, 401, 632-638, 664-668 |
| cashocs/io/mesh.py                                                                 |      235 |       32 |     86% |91, 146-150, 283-308, 334, 363-364, 397, 553-554, 558-560, 631, 664 |
| cashocs/io/output.py                                                               |       57 |        0 |    100% |           |
| cashocs/log.py                                                                     |      115 |       14 |     88% |114, 231-232, 255-262, 271, 275, 279 |
| cashocs/nonlinear\_solvers/\_\_init\_\_.py                                         |       11 |        0 |    100% |           |
| cashocs/nonlinear\_solvers/linear\_solver.py                                       |       35 |        7 |     80% |     88-96 |
| cashocs/nonlinear\_solvers/newton\_solver.py                                       |      196 |       26 |     87% |123-128, 166-169, 188-191, 229-231, 260, 265, 316-318, 358, 366-372, 387-388, 401, 403, 436-437 |
| cashocs/nonlinear\_solvers/picard\_solver.py                                       |       82 |        7 |     91% |54-57, 164-166, 174 |
| cashocs/nonlinear\_solvers/snes.py                                                 |      109 |       10 |     91% |114-119, 144-148, 217-219, 256 |
| cashocs/nonlinear\_solvers/ts.py                                                   |      183 |       29 |     84% |124-127, 132-137, 152, 157, 167-171, 180-183, 189, 218, 220, 233, 267-268, 304-310, 367-369, 460, 473 |
| cashocs/space\_mapping/\_\_init\_\_.py                                             |        3 |        0 |    100% |           |
| cashocs/space\_mapping/optimal\_control.py                                         |      379 |       42 |     89% |161, 260, 340-345, 444, 531-532, 662-664, 671-673, 700-734, 919, 948-950, 960-962, 976-977 |
| cashocs/space\_mapping/shape\_optimization.py                                      |      393 |       55 |     86% |163, 263, 335-336, 429, 506-507, 522-549, 672-673, 690-692, 698-700, 732-772, 944, 971-973, 983-985, 999-1000 |
| cashocs/verification.py                                                            |      123 |        3 |     98% |214-215, 247 |
|                                                                          **TOTAL** | **7414** |  **614** | **92%** |           |


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