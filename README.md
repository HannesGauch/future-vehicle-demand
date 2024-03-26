This repository contains tools to model future vehicle stocks and flows.
It has three subtools:
1) Data handling and consolidation for past vehicle ownership, gdp, population, population density, urban percentage, urban population density.
2) Computation of bootstrapped Gompertz curve fits to the consolidated data
3) Flexible stock models

The tool can make predictions such as these:

![Screenshot 2024-03-26 at 18 58 49](https://github.com/HannesGauch/future-vehicle-demand/assets/51442929/e0e6366a-844c-4a34-b9a2-f01fca3b2c00)


These show median prediction, interquartile range and 95% prediction intervals.

To get started run the script in Tests/vehicle_stock_model_test.ipynb.

Acknowledgements

This material has been produced under the Climate Compatible Growth programme, which is funded by UK aid from the UK government, and Chatham House. However, the views expressed herein do not necessarily reflect the UK government's official policies.
