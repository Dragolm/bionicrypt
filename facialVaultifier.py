import cv2
import dlib
import random
from itertools import combinations
import pickle
import hashlib

# Note: Install dlib and opencv: pip install dlib opencv-python
# Download shape_predictor_68_face_landmarks.dat from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Unzip and place in the same directory

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
PRIME = 2**61 - 1  # Large prime
DEGREE = 4  # Degree of polynomial, combinations of DEGREE+1 = 5, manageable
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

def eval_poly(coeffs, x, prime):
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % prime
    return result

def lock(points, secret, degree, prime):
    coeffs = [secret] + [random.randint(0, prime-1) for _ in range(degree)]
    print(coeffs)
    genuine_points = [(p, eval_poly(coeffs, p, prime)) for p in points]
    chaff_count = len(points) * 10
    chaff_points = []
    used_x = set(points)
    while len(chaff_points) < chaff_count:
        x = random.randint(0, prime-1)
        if x not in used_x:
            used_x.add(x)
            y = random.randint(0, prime-1)
            true_y = eval_poly(coeffs, x, prime)
            if y != true_y:
                chaff_points.append((x, y))
    vault = genuine_points + chaff_points
    print(vault)
    random.shuffle(vault)
    return vault, hashlib.sha256(str(secret).encode()).hexdigest()  # Store hash for verification

def compute_p_at_x(sub, x, prime):
    result = 0
    for i in range(len(sub)):
        x_i, y_i = sub[i]
        term = y_i
        for j in range(len(sub)):
            if i != j:
                x_j = sub[j][0]
                denom = (x_i - x_j) % prime
                denom_inv = pow(denom, prime-2, prime)
                numer = (x - x_j) % prime
                term = (term * numer * denom_inv) % prime
        result = (result + term) % prime
    return result

def unlock(points, vault, degree, prime, threshold=0.7):
    candidate_points = []
    for x in points:
        for vx, vy in vault:
            if vx == x:
                candidate_points.append((vx, vy))
                break
    num_points = len(candidate_points)
    if num_points < degree + 1:
        return None
    for sub in combinations(candidate_points, degree + 1):
        sub = list(sub)
        fit_count = 0
        for cx, cy in candidate_points:
            computed_y = compute_p_at_x(sub, cx, prime)
            if computed_y == cy:
                fit_count += 1
        if fit_count >= threshold * num_points:
            secret = compute_p_at_x(sub, 0, prime)
            return secret
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
    if mode == "enroll":
        image = cv2.imread("athul.png")
        if image is None:
            print("Capture failed")
        else:
            points = get_landmarks(image)
            if points is None:
                print("No face detected")
            else:
                secret = int(input("Enter secret key (integer): "))
                vault, h = lock(points, secret, DEGREE, PRIME)
                with open("vault.pkl", "wb") as f:
                    pickle.dump((vault, h), f)
                print("Enrollment complete. Vault saved.")
    elif mode == "verify":
        image = cv2.imread("athul.png")
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
                    recovered_secret = unlock(points, vault, DEGREE, PRIME)
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