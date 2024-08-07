# 2-Opt, to optimize the path to solve the Travelling salesman problem, TSP
def steep_improve_route(route, dists, max_iter):
    start_node = route[0]
    # Initialize savings, iter, etc.
    savings = 1
    iters = 0
    dimension = len(route)
    while iters < max_iter and savings > 0:
        savings = 0
        t2_sel = None
        t4_sel = None
        for t1_idx in range(dimension):
            t2_idx = (t1_idx + 1) % dimension
            for t4_idx in range(dimension):
                if t4_idx != t1_idx and t4_idx != t2_idx and (t4_idx + 1) % dimension != t1_idx:
                    t3_idx = (t4_idx + 1) % dimension
                    t1 = route[t1_idx]
                    t2 = route[t2_idx]
                    t3 = route[t3_idx]
                    t4 = route[t4_idx]
                    dis_diff = dists[t1][t2] + dists[t4][t3] - dists[t2][t3] - dists[t1][t4]
                    if dis_diff > savings:
                        savings = dis_diff
                        t2_sel = t2_idx
                        t4_sel = t4_idx
        if savings > 0:
            route = re_arrange_route(t2_sel, t4_sel, route)
        iters += 1
    # Reorganize the route to start and end at 0
    start_index = route.index(start_node)
    route = route[start_index:] + route[:start_index]
    return route


# Reorganize the route for `steep_improve_method()`
def re_arrange_route(t2_sel, t4_sel, route):
    if t2_sel < t4_sel:
        new_route = route.copy()
        # Reverse the elements between `t2_sel` and `t4_sel`
        new_route[t2_sel + 1:t4_sel] = new_route[t2_sel + 1:t4_sel][::-1]
        # Swap the elements at positions `t2_sel` and `t4_sel`
        new_route[t2_sel], new_route[t4_sel] = new_route[t4_sel], new_route[t2_sel]
    else:
        sub1 = route[t2_sel + 1:] + route[:t4_sel]
        sub1 = sub1[::-1]
        new_route = route[t4_sel + 1:t2_sel]
        new_route.append(route[t4_sel])
        new_route.extend(sub1)
        new_route.append(route[t2_sel])
    return new_route
