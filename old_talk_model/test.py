from scipy import stats

for _ in range(0, 30):
   norm = stats.norm.rvs(loc=1, scale=1, size=1)
   print(round(norm[0]))
