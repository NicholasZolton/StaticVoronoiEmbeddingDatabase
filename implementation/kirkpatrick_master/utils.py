import scipy


def between(a, b, c) -> bool:
    """Accepts 3 polygons.Point, return True iff b between a, c"""
    return (a.x < b.x < c.x or c.x < b.x < a.x) and \
        (a.y < b.y < c.y or c.y < b.y < a.y)


def ccw(a, b, c) -> int:
    """
    Advanced CCW method:
        abc is ccw: return 1
        
        abc is cw:  return -1
        
        abc are collinear and c between a, b: return 0
        
        abc are collinear and b between a, c: return 2
        
        abc are collinear and a between c, b: return -2
        
    :param a: Point
    :param b: Point
    :param c: Point
    """
    xp = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    if xp > 0:
        return 1
    if xp < 0:
        return -1

    if between(a, c, b):
        return 0
    if between(a, b, c):
        return 2
    if between(c, a, b):
        return -2


def intersect(a, b, c, d, closed=True) -> bool:
    """
    Return True if segment ab intersects segment cd
    
    :param a: polygons.Point
    
    :param b: polygons.Point
    
    :param c: polygons.Point
    
    :param d: polygons.Point
    
    :param closed: is the segment open or closed on ends
    
    :return: bool
    """
    # See example:
    #      B
    #     /
    # C--------D
    #   /
    #  A
    if closed:
        return ccw(a, b, c) != ccw(a, b, d) and ccw(a, c, d) != ccw(b, c, d)

    return intersect(a, b, c, d) and (a not in [c, d]) and (b not in [c, d])


def dist(a, b):
    diff = b - a
    return scipy.linalg.norm(diff.tuple())