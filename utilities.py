"""
General utility functions.
"""

rank = 0
def Put(*args, root: bool = False, **kwargs):
    """
    Function for convenient MPI printing. 
    
    Displays ID of printing process and includes flag 
    for only printing on the root process.
    Should be used exactly as print is used, but root=True
    will force only the root process to actually print
    """
    if root and rank != 0:
        return
    print(f"({rank})", end=" ")
    print(*args, **kwargs)
