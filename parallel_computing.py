
def uniform_distribute_tasks_across_cores(num_tasks, num_cores):
    range_parallel = [[] for d1 in range(num_cores)]
    for curr_core in range(num_cores):
        range_parallel[curr_core] = range(curr_core, num_tasks, num_cores)
    # print 'range parallel is ', range_parallel
    return range_parallel

