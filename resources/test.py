def insert(val, length, l):
    curPos = 0
    while curPos < length and l[curPos] <= val:
        curPos += 1
    i = length - 1
    while i >= curPos:
        l[i+1] = l[i]
        i -= 1
    l[curPos] = val
    return l

def main():
    a = [0 for i in range(10)]
    a = insert(5,0,a)
    a = insert(10,1,a)
    a = insert(15, 2, a)
    a = insert(3, 3, a)
    a = insert(-5, 4, a)
    print(a)

if __name__ == "__main__":
    main()