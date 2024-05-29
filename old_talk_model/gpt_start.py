import os
import shutil
import numpy as np


# mode = 0 : program start
# mode = 1 : Reuse data
mode = 2


if mode == 0:
    import gpt_graph

if mode == 1:
    import gpt_call
    import gpt_KL

if mode == 2:
    import gpt_graph2