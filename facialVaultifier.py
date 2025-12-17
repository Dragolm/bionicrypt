import cv2
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
import dlib
import random
from itertools import combinations
import pickle
import hashlib
import utilities as util
import evaluator

# Note: Install dlib and opencv: pip install dlib opencv-python
# Download shape_predictor_68_face_landmarks.dat from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Unzip and place in the same directory

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
PRIME = 2**521 - 1  # Large prime
QUANTIZE = 5  # Quantization step for tolerance

def get_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    points = []
    for i in range(68):
        x = landmarks.part(i).x // QUANTIZE
        y = landmarks.part(i).y // QUANTIZE
        point = x + y * (1000 // QUANTIZE)  # Assume image width <1000
        points.append(point)
    return sorted(set(points))  # Unique sorted list

def eval_poly(coeffs, p):
    result = coeffs[0]
    result = result*(p%evaluator.THE_KEY)
    return result

def lock(points, secret, key_len):
    max_point = max(points)
    min_point = min(points)
    # print(secret)
    coeffs = [secret]
    # print(coeffs)
    genuine_points = [(p, eval_poly(coeffs, p)) for p in points]
    chaff_count = len(points) * 5
    chaff_points = []
    used_x = set(points)
    while len(chaff_points) < chaff_count:
        x = random.randint(min_point, max_point)
        if x not in used_x:
            used_x.add(x)
            y = random.randint(int('1'+'0'*(key_len-1)), int('9'*key_len))
            true_y = eval_poly(coeffs, x)
            if y != true_y:
                chaff_points.append((x, y))
    vault = genuine_points + chaff_points
    random.shuffle(vault)
    return vault, hashlib.sha256(str(secret).encode()).hexdigest()  # Store hash for verification


def unlock(points, vault, prime, threshold=0.7):
    candidate_points = []
    for x in points:
        for vx, vy in vault:
            if vx == x:
                candidate_points.append((vx, vy))
                break
    key = candidate_points[0][1]//(candidate_points[0][0]%evaluator.THE_KEY)
    hits = 0
    for i in range(1, len(candidate_points)):
        try:
            temp_key = candidate_points[i][1]//(candidate_points[i][0]%evaluator.THE_KEY)
            if temp_key==key:
                hits+=1
        except Exception as e:
            print(e)
            print(candidate_points[i][0]) 
            print(candidate_points[i][1])
    if hits>len(candidate_points)-10:
        return key
    return None

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening camera")
        return None
    print("Press 'c' to capture")
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Capture', frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                cap.release()
                cv2.destroyAllWindows()
                return frame
        else:
            cap.release()
            cv2.destroyAllWindows()
            return None

if __name__ == "__main__":
    mode = input("Enter mode (enroll/verify): ").strip().lower()
    # For test case can be removed in production
    if input("Is this a test?(y/n)").strip().lower()=="y":
        image = cv2.imread("athul.png")
    else:
        image = capture_image()
    if mode in ["enroll", "e"]:
        if image is None:
            print("Capture failed")
        else:
            points = get_landmarks(image)
            if points is None:
                print("No face detected")
            else:
                # Generating a 256 bit private key
                private_key = ec.generate_private_key(ec.SECP256K1()).private_numbers().private_value
                util.filewriter(str(private_key), 'priv_key')
                vault, h = lock(points, private_key, len(str(private_key)))
                #Dumping the vault into a pickle file
                with open("vault.pkl", "wb") as f:
                    pickle.dump((vault, h), f)
                print("Enrollment complete. Vault saved.")
    elif mode in ["verify", "v"]:
        if image is None:
            print("Capture failed")
        else:
            points = get_landmarks(image)
            if points is None:
                print("No face detected")
            else:
                try:
                    with open("vault.pkl", "rb") as f:
                        vault, stored_hash = pickle.load(f)
                    util.writer("")
                    for j in vault:
                        util.appender(str(j))
                    recovered_secret = unlock(points, vault, PRIME)
                    if recovered_secret is not None:
                        recovered_hash = hashlib.sha256(str(recovered_secret).encode()).hexdigest()
                        if recovered_hash == stored_hash:
                            print("Verification successful. Secret key:", recovered_secret)
                        else:
                            print("Verification failed.",recovered_secret)
                    else:
                        print("Verification failed.")
                except FileNotFoundError:
                    print("No vault found. Enroll first.")
    else:
        print("Invalid mode.")