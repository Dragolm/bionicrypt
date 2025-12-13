import numpy as np
import os
from PIL import Image
from scipy.linalg import orth
from itertools import combinations
import pickle
import random

class FuzzyVaultFace:
    def __init__(self, M=20, D=8, w=2, tau=200, num_chaff=200, user_dependent=False):
        """
        Initialize Fuzzy Vault for Face Recognition
        
        Parameters:
        - M: Number of binary features (9-20)
        - D: Polynomial degree (default 8)
        - w: Window size for matching (1-5)
        - tau: Range for random vector generation
        - num_chaff: Number of chaff points
        - user_dependent: If True, use user-specific random matrices
        """
        self.M = M
        self.D = D
        self.w = w
        self.tau = tau
        self.num_chaff = num_chaff
        self.user_dependent = user_dependent
        self.N = 20  # PCA dimensions
        
        # PCA parameters
        self.mean_face = None
        self.eigenfaces = None
        
        # Global random matrices (for user-independent mode)
        self.global_Q1 = None
        self.global_Q2 = None
        self.global_r1 = None
        self.global_r2 = None
        
    def load_dataset(self, dataset_path, img_size=(92, 112)):
        """Load face dataset from directory structure"""
        images = []
        labels = []
        
        for person_id in range(1, 41):  # s1 to s40
            folder = f"s{person_id}"
            folder_path = os.path.join(dataset_path, folder)
            
            if not os.path.exists(folder_path):
                continue
                
            for img_num in range(1, 11):  # 1 to 10 images
                img_path = os.path.join(folder_path, f"{img_num}.pgm")
                if not os.path.exists(img_path):
                    continue
                    
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                img_array = np.array(img).flatten()
                images.append(img_array)
                labels.append(person_id - 1)  # 0-indexed
        
        return np.array(images), np.array(labels)
    
    def train_pca(self, images, n_components=20):
        """Train PCA on face images"""
        # Compute mean face
        self.mean_face = np.mean(images, axis=0)
        
        # Center the data
        centered = images - self.mean_face
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top n_components
        self.eigenfaces = eigenvectors[:, :n_components]
        
        return eigenvalues[:n_components]
    
    def extract_features(self, image):
        """Extract PCA features from image"""
        centered = image - self.mean_face
        features = self.eigenfaces.T @ centered
        return features
    
    def generate_random_matrices(self):
        """Generate random orthonormal matrices Q1, Q2 and random vectors r1, r2"""
        # Generate two random matrices and orthonormalize
        random_mat1 = np.random.randn(self.N, self.M)
        random_mat2 = np.random.randn(self.N, self.M)
        
        Q1 = orth(random_mat1)
        Q2 = orth(random_mat2)
        
        # Generate random vectors with elements in [0, tau]
        r1 = np.random.uniform(0, self.tau, self.M)
        r2 = np.random.uniform(0, self.tau, self.M)
        
        return Q1, Q2, r1, r2
    
    def binary_mapping(self, features, Q1, Q2, r1, r2):
        """Map face features to binary representation using 2D quantization"""
        # Compute R1 and R2
        R1 = Q1 * r1  # Broadcasting: each column multiplied by corresponding r1 element
        R2 = Q2 * r2
        
        # Compute Euclidean distances
        d1 = np.zeros(self.M)
        d2 = np.zeros(self.M)
        
        for i in range(self.M):
            d1[i] = np.linalg.norm(features - R1[:, i])
            d2[i] = np.linalg.norm(features - R2[:, i])
        
        # Find max distances for normalization
        max_d1 = np.max(d1)
        max_d2 = np.max(d2)
        
        # Quantize to 256 steps (8 bits)
        b1 = np.clip((d1 / (max_d1 + 1e-10) * 255).astype(int), 0, 255)
        b2 = np.clip((d2 / (max_d2 + 1e-10) * 255).astype(int), 0, 255)
        
        # Combine to 16-bit binary features
        binary_features = (b1.astype(np.uint16) << 8) | b2.astype(np.uint16)
        
        return binary_features
    
    def crc16(self, data):
        """Calculate CRC-16 using polynomial x^16 + x^15 + x^2 + 1"""
        crc = 0xFFFF
        poly = 0x8005  # x^16 + x^15 + x^2 + 1
        
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ poly
                else:
                    crc = crc << 1
                crc &= 0xFFFF
        
        return crc
    
    def generate_key(self):
        """Generate random 128-bit key"""
        return np.random.randint(0, 256, 16, dtype=np.uint8)
    
    def encode_vault(self, binary_features, key):
        """Encode fuzzy vault with key and binary features"""
        # Add CRC-16 to key
        crc = self.crc16(key)
        key_with_crc = np.concatenate([key, [(crc >> 8) & 0xFF, crc & 0xFF]])
        
        # Convert to 9 16-bit coefficients for degree-8 polynomial
        coeffs = []
        for i in range(9):
            c = (int(key_with_crc[i*2]) << 8) | int(key_with_crc[i*2 + 1])
            coeffs.append(c)
        coeffs = np.array(coeffs)
        
        # Evaluate polynomial at feature points (genuine set)
        genuine_set = []
        for x in binary_features:
            y = np.polyval(coeffs[::-1], x)  # polyval expects highest degree first
            genuine_set.append((int(x), int(y) & 0xFFFFFFFF))  # Keep 32-bit
        
        # Generate chaff points
        chaff_set = []
        all_x = set(binary_features)
        
        for _ in range(self.num_chaff):
            # Generate random x not in genuine set
            x = np.random.randint(0, 65536)
            while x in all_x:
                x = np.random.randint(0, 65536)
            all_x.add(x)
            
            # Generate random y that doesn't lie on polynomial
            y_genuine = np.polyval(coeffs[::-1], x)
            y = np.random.randint(0, 2**32)
            while y == (int(y_genuine) & 0xFFFFFFFF):
                y = np.random.randint(0, 2**32)
            
            chaff_set.append((x, y))
        
        # Combine and shuffle
        vault = genuine_set + chaff_set
        random.shuffle(vault)
        
        return vault, key
    
    def decode_vault(self, vault, binary_features_auth):
        """Decode vault and retrieve key"""
        # Find candidate points with windowing
        candidates = []
        
        for bf in binary_features_auth:
            # Split into x and y coordinates (8-bit each)
            x_auth = (bf >> 8) & 0xFF
            y_auth = bf & 0xFF
            
            for x_vault, y_vault in vault:
                # Split vault point
                x_v = (x_vault >> 8) & 0xFF
                y_v = x_vault & 0xFF
                
                # Check if within window
                if (abs(x_auth - x_v) <= self.w and 
                    abs(y_auth - y_v) <= self.w):
                    candidates.append((x_vault, y_vault))
        
        # Remove duplicates
        candidates = list(set(candidates))
        
        K = len(candidates)
        if K < self.D + 1:
            return None
        
        # Try all combinations of D+1 points
        for combo in combinations(candidates, self.D + 1):
            x_vals = np.array([p[0] for p in combo])
            y_vals = np.array([p[1] for p in combo])
            
            # Lagrange interpolation
            try:
                coeffs = self.lagrange_interpolation(x_vals, y_vals)
                
                # Convert coefficients back to key
                key_with_crc = []
                for c in coeffs:
                    key_with_crc.append((c >> 8) & 0xFF)
                    key_with_crc.append(c & 0xFF)
                
                key_with_crc = np.array(key_with_crc[:18], dtype=np.uint8)
                
                # Verify CRC
                key = key_with_crc[:16]
                crc_stored = (int(key_with_crc[16]) << 8) | int(key_with_crc[17])
                crc_computed = self.crc16(key)
                
                if crc_stored == crc_computed:
                    return key
            except:
                continue
        
        return None
    
    def lagrange_interpolation(self, x_vals, y_vals):
        """Lagrange interpolation to reconstruct polynomial coefficients"""
        n = len(x_vals)
        coeffs = np.zeros(n, dtype=np.int64)
        
        for i in range(n):
            # Build Lagrange basis polynomial
            basis = np.array([1.0])
            for j in range(n):
                if i != j:
                    # Multiply by (x - x_j) / (x_i - x_j)
                    denom = x_vals[i] - x_vals[j]
                    if denom == 0:
                        raise ValueError("Duplicate x values")
                    basis = np.polymul(basis, [1, -x_vals[j]])
                    basis = basis / denom
            
            # Multiply by y_i and add to result
            coeffs = coeffs + y_vals[i] * basis
        
        # Convert to integer coefficients
        coeffs = np.round(coeffs).astype(np.int64)
        
        # Take first D+1 coefficients and convert to 16-bit
        return (coeffs[:self.D + 1] & 0xFFFF).astype(np.uint16)
    
    def enroll(self, images, person_id):
        """Enroll user and create vault"""
        # Extract features from all enrollment images
        features_list = []
        for img in images:
            feat = self.extract_features(img)
            features_list.append(feat)
        
        # Use mean of features
        mean_features = np.mean(features_list, axis=0)
        
        # Generate or use random matrices
        if self.user_dependent:
            Q1, Q2, r1, r2 = self.generate_random_matrices()
        else:
            if self.global_Q1 is None:
                # Initialize global matrices
                self.global_Q1, self.global_Q2, self.global_r1, self.global_r2 = \
                    self.generate_random_matrices()
            Q1, Q2, r1, r2 = self.global_Q1, self.global_Q2, self.global_r1, self.global_r2
        
        # Binary mapping
        binary_features = self.binary_mapping(mean_features, Q1, Q2, r1, r2)
        
        # Generate key and create vault
        key = self.generate_key()
        vault, key = self.encode_vault(binary_features, key)
        
        # Store enrollment data
        enrollment_data = {
            'vault': vault,
            'key': key,  # Store for verification (in real system, this would not be stored)
            'Q1': Q1,
            'Q2': Q2,
            'r1': r1,
            'r2': r2,
            'person_id': person_id
        }
        
        return enrollment_data
    
    def authenticate(self, image, enrollment_data):
        """Authenticate user and retrieve key"""
        # Extract features
        features = self.extract_features(image)
        
        # Get random matrices from enrollment
        Q1 = enrollment_data['Q1']
        Q2 = enrollment_data['Q2']
        r1 = enrollment_data['r1']
        r2 = enrollment_data['r2']
        
        # Binary mapping
        binary_features = self.binary_mapping(features, Q1, Q2, r1, r2)
        
        # Decode vault
        retrieved_key = self.decode_vault(enrollment_data['vault'], binary_features)
        
        return retrieved_key


# Example usage
if __name__ == "__main__":
    # Initialize system
    print("Initializing Fuzzy Vault Face Recognition System...")
    
    # User-dependent mode for better accuracy
    system = FuzzyVaultFace(M=20, D=8, w=3, user_dependent=True)
    
    # Load dataset
    dataset_path = "face_dataset"  # Change to your dataset path
    print(f"Loading dataset from {dataset_path}...")
    
    try:
        images, labels = system.load_dataset(dataset_path)
        print(f"Loaded {len(images)} images from {len(np.unique(labels))} persons")
        
        # Train PCA on first 5 images per person (gallery set)
        print("Training PCA...")
        train_images = images[::2][:200]  # First 5 images of each person
        system.train_pca(train_images, n_components=20)
        
        # Enroll first person (using first 5 images)
        person_id = 0
        enrollment_images = images[labels == person_id][:5]
        print(f"\nEnrolling person {person_id + 1}...")
        enrollment_data = system.enroll(enrollment_images, person_id)
        print(f"Vault created with {len(enrollment_data['vault'])} points")
        print(f"Generated key: {enrollment_data['key'].hex()}")
        
        # Test authentication with same person (image 6)
        test_image_genuine = images[labels == person_id][5]
        print("\nTesting with genuine user (same person, different image)...")
        retrieved_key = system.authenticate(test_image_genuine, enrollment_data)
        
        if retrieved_key is not None:
            print(f"✓ Authentication successful!")
            print(f"Retrieved key: {retrieved_key.hex()}")
            print(f"Keys match: {np.array_equal(retrieved_key, enrollment_data['key'])}")
        else:
            print("✗ Authentication failed - could not retrieve key")
        
        # Test with imposter (different person)
        imposter_id = 1
        test_image_imposter = images[labels == imposter_id][0]
        print("\nTesting with imposter (different person)...")
        retrieved_key_imp = system.authenticate(test_image_imposter, enrollment_data)
        
        if retrieved_key_imp is None:
            print("✓ Correctly rejected imposter")
        else:
            print(f"✗ Imposter authenticated (should not happen)")
            
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure your dataset is structured as: face_dataset/s1/1.pgm, etc.")