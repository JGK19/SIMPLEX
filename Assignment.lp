\* Assignment *\
Minimize
total_cost_for_jobs: 4 person_0_does_job_0 + 10 person_1_does_job_0
 + 4 person_1_does_job_1 + 3 person_1_does_job_2 + 7 person_2_does_job_0
 + 5 person_2_does_job_1 + 5 person_2_does_job_2
Subject To
_C1: person_0_does_job_0 + person_0_does_job_1 + person_0_does_job_2 = 1
_C2: person_1_does_job_0 + person_1_does_job_1 + person_1_does_job_2 = 1
_C3: person_2_does_job_0 + person_2_does_job_1 + person_2_does_job_2 = 1
_C4: person_0_does_job_0 + person_1_does_job_0 + person_2_does_job_0 = 1
_C5: person_0_does_job_1 + person_1_does_job_1 + person_2_does_job_1 = 1
_C6: person_0_does_job_2 + person_1_does_job_2 + person_2_does_job_2 = 1
Binaries
person_0_does_job_0
person_0_does_job_1
person_0_does_job_2
person_1_does_job_0
person_1_does_job_1
person_1_does_job_2
person_2_does_job_0
person_2_does_job_1
person_2_does_job_2
End
