rank = 0
def Put(*args, root: bool = False, **kwargs):
    if root and rank != 0:
        return
    print(f"({rank})", end=" ")
    print(*args, **kwargs)
