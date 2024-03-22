def addTb(tb1, tb2):
    """
    Add up two tight-binding models together.
    
    Parameters:
    -----------
    tb1 : dict
        Tight-binding model.
    tb2 : dict
        Tight-binding model.
    
    Returns:
    --------
    dict
        Sum of the two tight-binding models.
    """
    return {k: tb1.get(k, 0) + tb2.get(k, 0) for k in frozenset(tb1) | frozenset(tb2)}
