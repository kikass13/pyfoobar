



def iterative_topological_sort(graph, start):
    seen = set()
    stack = []    # path variable is gone, stack and order are new
    order = []    # order will be in reverse order at first
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v) # no need to append to path any more
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]: # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    return stack + order[::-1]   # new return value!

graph = {
    'a': ['b', 'c'],
    'b': ['d'],
    'c': ['d'],
    'd': ['e'],
    'e': []
}
print(iterative_topological_sort(graph, 'a'))