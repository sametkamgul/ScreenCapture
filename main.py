import time
import cv2 as cv
import mss
import numpy
from flask import Flask, render_template, Response
from win32api import GetSystemMetrics


app = Flask(__name__)


def stream_video(frame):
    while True:
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def screen_capture():
    cv.namedWindow("Colorbars")
    cv.createTrackbar("GrayColorRange-Start", "Colorbars", 0, 255, nothing)
    cv.createTrackbar("GrayColorRange-End", "Colorbars", 0, 255, nothing)
    with mss.mss() as sct:
        # Part of the screen to capture
        screen_width = GetSystemMetrics(0)
        screen_height = GetSystemMetrics(1)
        print("INFO:", "screen size:", screen_width, "x", screen_height)
        monitor = {"top": 0, "left": 0, "width": screen_width, "height": screen_height}

        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))

            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

            # get value from trackbar
            lower_gray_color_val = cv.getTrackbarPos("GrayColorRange-Start", "Colorbars")
            upper_gray_color_val = cv.getTrackbarPos("GrayColorRange-End", "Colorbars")

            print(lower_gray_color_val, upper_gray_color_val)

            # finding squares
            result = find_squares(img, lower_gray_color_val, upper_gray_color_val)
            #cv.imshow("squares", result)

            #scaling the image for performance
            dsize = (int(result.shape[1] / 3), int(result.shape[0] / 3))
            result = cv.resize(result, dsize)

            ret, buffer = cv.imencode('.jpg', result)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

            # Display the original picture
            #cv.imshow("OpenCV/Numpy normal", img)

            print("fps: {}".format(1 / (time.time() - last_time)))



            # Press "q" to quit
            if cv.waitKey(1) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break


def find_squares(img, lower_range, upper_range):
    """
    finds the square or rectangle with a certain range of width and height
    :param img:
    :return: img
    """




    imgGry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thrash = cv.threshold(imgGry, lower_range, upper_range, cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    square_list = []

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.05 * cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)
            if w >=100 and w <= 1600 and h >= 100 and h <= 1600:

                if [x, y, w, h] in square_list:
                    #print("exists")
                    pass
                else:
                    #print("not exists")
                    square_list.append([x, y, w, h])
                cv.drawContours(img, [approx], 0, (0, 255, 0), 5)
                aspectRatio = float(w) / h
                #print(aspectRatio)
                if aspectRatio >= 0.95 and aspectRatio < 1.05:
                    cv.putText(img, "detected-square", (x, y + 100), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                else:
                    cv.putText(img, "detected-rectangel", (x, y + 100), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            else:
                pass
        else:
            pass
    print(len(square_list), square_list)    # listing the rect areas
    return img

def nothing(x):
    pass


@app.route('/')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(screen_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)