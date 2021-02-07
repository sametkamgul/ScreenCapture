import time
import cv2 as cv
import mss
import numpy
from flask import Flask, render_template, Response


app = Flask(__name__)


def stream_video(frame):
    while True:
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def screen_capture():
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 0, "left": 0, "width": 1200, "height": 800}

        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))

            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)

            # finding squares
            result = find_squares(img)
            cv.imshow("squares", result)

            ret, buffer = cv.imencode('.jpg', result)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

            # Display the original picture
            #cv.imshow("OpenCV/Numpy normal", img)

            print("fps: {}".format(1 / (time.time() - last_time)))

            # Press "q" to quit
            if cv.waitKey(25) & 0xFF == ord("q"):
                cv.destroyAllWindows()
                break


def find_squares(img):
    """
    finds the square or rectangle with a certain range of width and height
    :param img:
    :return: img
    """
    imgGry = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thrash = cv.threshold(imgGry, 240, 255, cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv.findContours(thrash, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)
            if w >=150 and w <= 500 and h >= 150 and h <= 500:
                cv.drawContours(img, [approx], 0, (0, 255, 0), 5)
                aspectRatio = float(w) / h
                print(aspectRatio)
                if aspectRatio >= 0.95 and aspectRatio < 1.05:
                    cv.putText(img, "detected-square", (x, y + 100), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                else:
                    cv.putText(img, "detected-rectangel", (x, y + 100), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            else:
                pass
        else:
            pass
    return img


@app.route('/')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(screen_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)