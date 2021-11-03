import tsp
import sqtsp

MAX_CITIES=30
MIN_CITIES=5

print("comparing solutions")

wins_tsp =0
wins_sqtsp=0

for i in range(MIN_CITIES,MAX_CITIES):
    tsp_sol = tsp.execute(i)
    sqtsp_sol = sqtsp.execute(i)

    if(tsp_sol==sqtsp_sol):
        print(f"draw with {i} cities")
    elif tsp_sol < sqtsp_sol:
        print(f"my tweak wins with {i} cities")
        wins_tsp+=1
    else:
        print(f"tweak developed in class wins with {i} cities")
        wins_sqtsp+=1

print(f"my solution won {wins_tsp} times, solution developed in class wins {wins_sqtsp} times")
