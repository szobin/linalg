from intervals import SegType, Segment, SegmentSet

s1 = Segment(SegType.EMPTY)
print(s1)

s2 = Segment(SegType.OPEN, 1, 5)
print(s2)

s3 = Segment(SegType.OPEN, 4, 8)
print(s3)

s = s2 + s3
print(s)

ss = SegmentSet()
ss.append(s)
