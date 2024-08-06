#solution 1 
def count_covered_lines(lines, pos_):
    covered_pos = set(tuple(pos) for pos in pos_)

    def cal_points(a, b, c, d):
       points = []
       if a == c :
         if b > d:
            for y in range(d, b + 1):
                points.append(a, y)
         if b < d:
            for y in range(b, d + 1):
                points.append(a, y)
       if b == d:
         if a < c:
            for x in range (a, c + 1):
                points.append(x, b)
         if a > c:
            for x in range(c, a+1):
                points.append(x, b)
       elif abs(a-c) == abs(b-d):
         if a < c and b < d:
            points = [(a + i, b + i) for i in range(c - a + 1)]
         elif a < c and b > d:
            points = [(a + i, b - i) for i in range(c - a + 1)]
         elif a > c and b < d:
            points = [(a - i, b + i) for i in range(a - c + 1)]
         elif a > c and b > d :
            points = [(a - i, b - i) for i in range(a - c + 1)]
       return points 

    def covered(x, y):
      return (x, y) in covered_pos

    wqfg = 0
    for l in lines:
       a, b, c, d = l
       points = cal_points(a, b, c, d)
       if all(covered(x, y) for (x, y) in points):
           wqfg += 1
    return wqfg



#solution 2 
def cal(a, b, c, d):
   
   p = []
   slope = abs(d - b) > abs(a - c)

   if slope:
      a, b = b, a
      c, d = d, c
    
   if a > c:
      a, c = c, a
      b, d = d, b

   d_x = c - a
   d_y = abs(d - b)

   e = d_x / 2
   y_step = 1 if b < d else -1
   y = b
    
   for x in range(a, c + 1):
      if slope:
         p.append((y, x))
      else:
         p.append((x, y))

      e -= d_y
      if e < 0 :
         y += y_step
         e += d_x

   return p

def solution2(a, b, c, d):
    
    points = cal(a, b, c, d)

    covered = set()

    for i in range(1, len(points) - 1):
       covered.add(points[i])

    return list(covered)

