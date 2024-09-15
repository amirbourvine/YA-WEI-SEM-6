import sys


if len(sys.argv) > 1:
    task_id = int(sys.argv[1]) 
    n_tasks = int(sys.argv[2])
else:
    task_id = 0  
    n_tasks = 1


n_tasks=2


for i in range(10):
    for j in range(5):

        combined_idx = i*10 + j
        if combined_idx not in range(task_id,10*5,n_tasks):            
            continue

        print(combined_idx)
