import numpy as np 

def error_rate(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]
 
def bb(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou
# def wer(r, h):
#     """
#     Calculation of WER with Levenshtein distance.

#     Works only for iterables up to 254 elements (uint8).
#     O(nm) time ans space complexity.

#     Parameters
#     ----------
#     r : list
#     h : list

#     Returns
#     -------
#     int

#     Examples
#     --------
#     >>> wer("who is there".split(), "is there".split())
#     1
#     >>> wer("who is there".split(), "".split())
#     3
#     >>> wer("".split(), "who is there".split())
#     3
#     """
#     # initialisation
#     d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
#     d = d.reshape((len(r)+1, len(h)+1))
#     for i in range(len(r)+1):
#         for j in range(len(h)+1):
#             if i == 0:
#                 d[0][j] = j
#             elif j == 0:
#                 d[i][0] = i

#     # computation
#     for i in range(1, len(r)+1):
#         for j in range(1, len(h)+1):
#             if r[i-1] == h[j-1]:
#                 d[i][j] = d[i-1][j-1]
#             else:
#                 substitution = d[i-1][j-1] + 1
#                 insertion    = d[i][j-1] + 1
#                 deletion     = d[i-1][j] + 1
#                 d[i][j] = min(substitution, insertion, deletion)

#     return d[len(r)][len(h)]


def iou(a,b):

    def intersection(a,b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[0]+a[2], b[0]+b[2])
        y2 = min(a[1]+a[3], b[1]+b[3])
        if x1 < x2 and y1 < y2:
            return (x1, y1, x2 - x1, y2 - y1)
        else:
            return (0, 0, 0, 0)

    def union(a,b):
        x1 = min(a[0], b[0])
        y1 = min(a[1], b[1])
        x2 = max(a[0]+a[2], b[0]+b[2])
        y2 = max(a[1]+a[3], b[1]+b[3])
        if x1 < x2 and y1 < y2:
            return (x1, y1, x2 - x1, y2 - y1)

    inter = intersection(a,b)
    uni = union(a,b)
    _iou = (inter[2]*inter[3])/(uni[2]*uni[3])


    return _iou
