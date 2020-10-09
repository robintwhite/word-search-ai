from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pytesseract
import numpy as np
import cv2
import os
import progressbar
import pickle


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


def get_pixel_coords(seq_dict, pixel_arr):
    ''' return the pixel x,y position of center of
    box for 1st and last letter. Used to click to complete game
    pixel_arr: array of centroid positions of letters in pixels.
               Need to add offset from screenshot
    '''
    pass


screenshot_dir = 'images'
custom_config = r'--psm 3 --oem 3'

words_location0, words_location1 = (980, 325), (1180, 940)
letters_location0, letters_location1 = (350, 305), (975, 930)

print("[INFO] loading pre-trained network...")
model = load_model('models')
# Hydrate the serialized objects.
with open('label-binarizer.pkl', 'rb') as f:
    lb = pickle.load(f)

widgets = ["Processing image: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]

for entry in os.scandir(screenshot_dir):
    labels = []
    if entry.path.endswith(".png") and entry.is_file():
        print('[INFO] assigning characters to image...')
        im = cv2.imread(entry.path)
        # letters grid (350, 305), (1060, 880)
        letters_grid = im[letters_location0[1]:letters_location1[1], letters_location0[0]:letters_location1[0]]
        gray = cv2.cvtColor(letters_grid.astype('uint8'), cv2.COLOR_BGR2GRAY)
        # gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh = thresh[1]
        conn = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_16U)
        stats = conn[2][1:] # stat matrix, #0 is bckgrnd
        cents = conn[3][1:] # centroids
        pad = 8

        words_crop = im[words_location0[1]:words_location1[1], words_location0[0]:words_location1[0]]
        words_gray = cv2.cvtColor(words_crop.astype('uint8'), cv2.COLOR_BGR2GRAY)

        txt = pytesseract.image_to_string(words_gray, config=custom_config)
        txt = clean_string(txt)
        target_words = [word for word in txt.splitlines()]

        pbar = progressbar.ProgressBar(maxval=len(stats),
                                       widgets=widgets).start()

        for i, (x, y, w, h, a) in enumerate(stats): #0 is bckgrnd
            # x,y,w,h,a = obj
            cent = tuple(list(map(int, cents[i])))
            cv2.rectangle(letters_grid, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 255, 0), 2)
            cv2.circle(letters_grid, cent, 5, (200, 150, 0), 1)
            # letter = thresh[y - pad:y + h + pad, x - pad:x + w + pad]
            roi = gray[y - pad:y + h + pad, x - pad:x + w + pad]
            image = cv2.resize(roi, (28, 28))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            pred = model.predict(image)
            pred = np.where(pred == np.max(pred, axis=1), 1, 0)
            label = lb.inverse_transform(pred).tolist()[0]
            labels.append(label)
            cv2.putText(letters_grid, label, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            pbar.update(i)
        cv2.imshow("ROI", letters_grid)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

        pbar.finish()

    letters_array = np.array(labels)
    dim = int(np.sqrt(len(letters_array)))
    letters_array = np.reshape(letters_array, (dim, dim))

    cents_array = np.array(cents).astype(int)
    cents_array = np.reshape(cents_array, (dim, dim, 2))
    print(cents_array)

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
                        # need to add letters_location0 to offsite for actual screen pixel coords
    print(word_locations_dict)
    break  # just run 1 for testing
