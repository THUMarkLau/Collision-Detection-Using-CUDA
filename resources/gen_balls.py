import numpy.random as r

BALL_NUM = 2000

def main():
    f = open("balls.txt", "w")

    f.write("%d\n"%(BALL_NUM))
    balls = set()
    while len(balls) < BALL_NUM:
        ball = [r.random() * 4.5 - 2.25, r.random() * 3.0, r.random() * 4.5 - 1.5]
        balls.add(tuple(ball))
    for ball in balls:
        f.write("%.3f %.3f %.3f\n"%(ball[0], ball[1], ball[2]))

    f.close()

if __name__ == "__main__":
    main()