import cv2

qr1 = cv2.imread('../test_pictures/qr-1.png')
cv2.imwrite('qr-1-org.png', qr1)
qr1_rgb = cv2.cvtColor(qr1, cv2.COLOR_BGR2RGB)
cv2.imwrite('qr-1-rgb.png', qr1_rgb)
qr_gray = cv2.cvtColor(qr1, cv2.COLOR_RGB2GRAY)
cv2.imwrite('qr-q-gray.png', qr_gray)
