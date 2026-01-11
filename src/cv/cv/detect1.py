import rclpy
from rclpy.node import Node
from example_interfaces.msg import Float32
import imutils
import numpy as np
import cv2
# Substitua pela distância focal que você obteve
FOCAL_LENGTH = 1460.1 # ← ajuste esse valor com base no seu resultado
KNOWN_WIDTH = 9.5    # cm

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

class Detect1Node(Node): 
    def __init__(self):
        super().__init__("detect1") 
        self.pubx = self.create_publisher(Float32, 'cam1/dist/x', 10)


    def process_frame(self, frame):
        camera = cv2.VideoCapture(2)
        if not camera.isOpened():
            print("Erro: não foi possível acessar a câmera.")
            exit()
        cv2.namedWindow("measure distance", cv2.WINDOW_NORMAL)  # ← ADICIONE AQUI
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=600)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, lower_red, upper_red)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                object_width = rect[1][0]

                if object_width > 0:
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / object_width
                    cv2.putText(frame, f"Distancia: {distance:.2f} cm", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                box = cv2.boxPoints(rect)
                box = box.astype(int)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            cv2.imshow("Medir Distancia", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        distx = Float32()
       
        distx.data = distance
       
        self.pubx.publish(distx)
        
        self.get_logger().info("STEP 7: published")
        camera.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = Detect1Node() # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()


