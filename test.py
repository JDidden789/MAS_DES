import itertools
import random

no_machine_rules = 2
no_job_rules = 2
no_seq_rules = 9

# for (f, e, d) in itertools.product(range(no_seq_rules), range(no_job_rules), range(no_machine_rules)):
#     print(d, e, f)

random.seed(10)
for i in range(11):
    print(random.expovariate(0.5))
