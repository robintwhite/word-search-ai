from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pytesseract
import keyboard
import pyautogui
import cv2
import numpy as np
import progressbar
import pickle
import os
import copy
import time


def getXY(ind, W):
    return ind // W, ind % W


def get_neighbors(x, y, W, H, index_arr):
    wlim = W - 1
    hlim = H - 1
    n1x, n1y = max((x - 1), 0), max((y - 1), 0)
    n2x, n2y = max((x - 1), 0), (y + 0)
    n3x, n3y = max((x - 1), 0), min((y + 1), wlim)
    n4x, n4y = (x + 0), max((y - 1), 0)
    n5x, n5y = (x + 0), min((y + 1), wlim)
    n6x, n6y = min((x + 1), hlim), max((y - 1), 0)
    n7x, n7y = min((x + 1), hlim), (y + 0)
    n8x, n8y = min((x + 1), hlim), min((y + 1), wlim)

    neighbors = [index_arr[n1x, n1y], index_arr[n2x, n2y], index_arr[n3x, n3y], index_arr[n4x, n4y],
                 index_arr[n5x, n5y], index_arr[n6x, n6y], index_arr[n7x, n7y], index_arr[n8x, n8y]]
    neighbors = set(neighbors)
    p = index_arr[x, y]
    try:
        neighbors.remove(p)
        return neighbors
    except:
        return neighbors


def getContinuingSeq(spos, epos, seq):
  (x1, x0), (y1, y0) = list(zip(epos, spos))
  v = (x1 - x0, y1 - y0)
  new_seq_pos = []
  for i in range(len(seq)):
    new_seq_pos.append([spos[0] + v[0]*i, spos[1] + v[1]*i])
  return new_seq_pos


def getLetters(seq, letter_arr):
  letters = [letter_arr[posx, posy] for posx, posy in seq]
  return letters


def getIndices(seq, index_arr):
  indices = [index_arr[posx, posy] for posx, posy in seq]
  return indices


def clean_string(string):
    tmp = ''
    for line in string:
        tmp += line.strip(' ').replace('-', '').replace('.', '')
    tmp = tmp.replace("'", '"')
    tmp = os.linesep.join([s for s in tmp.splitlines() if s])
    return tmp


custom_config = r'--psm 3 --oem 3'
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


print("[INFO] loading pre-trained network...")
model = load_model('models')
with open('label-binarizer.pkl', 'rb') as f:
    lb = pickle.load(f)

widgets = ["Processing image: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]


print('[INFO] Hit "s" when ready to take screenshot. Do not click off until prompted.')
keyboard.wait('s')

img = np.array(pyautogui.screenshot())
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
draw_img = copy.deepcopy(img)

gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel3)
nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
# ignore background
# keep only largest object (grid)
new_img = np.zeros_like(gray)
max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
new_img[labels == max_label] = 255
#print(stats[max_label, :])
minLineLength = 100
maxLineGap = 2
# lines = cv2.HoughLinesP(image=new_img, rho=3, theta=np.pi / 180, threshold=100, lines=None, minLineLength=minLineLength, maxLineGap=maxLineGap)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

bx1, by1, bw, bh, _ = stats[max_label, :]
# cv2.rectangle(draw_img, (bx1, by1), (bx1+bw, by1+bh), (0,50,255), 2)
ww = int(bw*(1.3))
# Grid outline
# cv2.rectangle(draw_img, (bx1+bw + 1, by1), (bx1 + ww, by1 + bh), (0, 255, 255), 2)

new_img_inv = 255-new_img
new_img_inv = cv2.morphologyEx(new_img_inv, cv2.MORPH_ERODE, kernel5)
nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(new_img_inv, connectivity=4)
print(f'Grid size (must be square): {int(np.sqrt(nb_components - 2))}')

for i, (x, y, w, h, a) in enumerate(stats[2:]):  # 0 is bckgrnd
    # x,y,w,h,a = obj
    cent = tuple(list(map(int, centroids[i+2])))
    cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.circle(draw_img, cent, 5, (200, 150, 0), 1)
# cv2.imshow("img", draw_img)
# if cv2.waitKey(0) & 0xFF == ord("q"):
#     cv2.destroyAllWindows()

pad = 8

letters_grid = img[by1-pad: by1 + bh + pad, bx1 - pad: bx1 + bw + pad]
letters_gray = cv2.cvtColor(letters_grid.astype('uint8'), cv2.COLOR_BGR2GRAY)
words_crop = gray[by1: by1 + bh, bx1 + bw + 1: bx1 + ww]
# cv2.imshow("grid", letters_grid)
# cv2.imshow("words", words_crop)
# if cv2.waitKey(0) & 0xFF == ord("q"):
#     cv2.destroyAllWindows()


thresh = cv2.threshold(letters_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
thresh = thresh[1]
# cv2.imshow("thresh", thresh)
# if cv2.waitKey(0) & 0xFF == ord("q"):
#     cv2.destroyAllWindows()
conn = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_16U)
stats = conn[2][1:] # stat matrix, #0 is bckgrnd
cents = conn[3][1:] # centroids


txt = pytesseract.image_to_string(words_crop, config=custom_config)
txt = clean_string(txt)
target_words = [word for word in txt.splitlines()]
print('')
print(f'[INFO] Found words as targets: {target_words}')

pbar = progressbar.ProgressBar(maxval=len(stats),
                               widgets=widgets).start()

labels = []
for i, (x, y, w, h, a) in enumerate(stats): #0 is bckgrnd
    # x,y,w,h,a = obj
    cent = tuple(list(map(int, cents[i])))
    # cv2.rectangle(letters_grid, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2)
    # cv2.circle(letters_grid, cent, 5, (200, 150, 0), 1)
    # letter = thresh[y - pad:y + h + pad, x - pad:x + w + pad]
    roi = letters_gray[y - pad:y + h + pad, x - pad:x + w + pad]
    image = cv2.resize(roi, (28, 28))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    pred = np.where(pred == np.max(pred, axis=1), 1, 0)
    label = lb.inverse_transform(pred).tolist()[0]
    labels.append(label)
    cv2.putText(letters_grid, label, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    pbar.update(i)
# cv2.imshow("ROI", letters_grid)
# if cv2.waitKey(0) & 0xFF == ord("q"):
#     cv2.destroyAllWindows()
#
pbar.finish()

letters_array = np.array(labels)
dim = int(np.sqrt(len(letters_array)))
letters_array = np.reshape(letters_array, (dim, dim))

cents_array = np.array(cents).astype(int)
cents_array = np.reshape(cents_array, (dim, dim, 2))
# print(cents_array)

W,H = letters_array.shape

# flatten returns copy, ravel returns view
index_array = np.arange(0, len(letters_array.ravel()), 1, dtype=np.uint8).reshape((W,H))

print('[INFO] checking possible words...')
word_locations_dict = {}
for seq in target_words:
    seq_list = list(seq)
    first_letter_positions = index_array[np.where(letters_array == seq_list[0])]
    for s in first_letter_positions:
        x0, y0 = getXY(s, W)
        neighbors = get_neighbors(x0, y0, W, H, index_array)
        for n in neighbors:
            x, y = getXY(n, W)
            pot_seq_pos = getContinuingSeq((x0, y0), (x, y), seq_list)
            if all(item >= 0 and item < W and item < H for sublist in pot_seq_pos for item in sublist):
                pot_seq = getLetters(pot_seq_pos, letters_array)
                if pot_seq == seq_list:
                    matched_seq = pot_seq
                    matched_seq_indices = getIndices(pot_seq_pos, index_array)
                    # print(pot_seq_pos)
                    # print(matched_seq)
                    # print(matched_seq_indices)

                    word_locations_dict[seq] = [cents_array[m,n].tolist()
                                                for m,n in pot_seq_pos]

#print(word_locations_dict)

# need to add letters_location0 to offsite for actual screen pixel coords
for key in word_locations_dict:
    print(key)
    print(word_locations_dict[key])
    time.sleep(0.05)
    # by1, bx1
    startx, starty = word_locations_dict[key][0][0]+bx1, word_locations_dict[key][0][1]+by1
    endx, endy = word_locations_dict[key][-1][0]+bx1, word_locations_dict[key][-1][1]+by1
    #print(startx, starty)
    #print(endx, endy)
    # pyautogui.position(startx, starty)
    # pyautogui.position(endx, endy)
    pyautogui.click(startx, starty, clicks=1, interval=1)
    time.sleep(0.05)
    pyautogui.click(endx, endy, clicks=1, interval=1)

print('[INFO] Done.')
