import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

class MemoryGame:
    """Main class for Memory card game vision system"""
    
    def __init__(self):
        self.cards = {}  # card_id: {bbox, state, region}
        self.current_player = 1
        self.player_scores = {1: 0, 2: 0}
        self.turned_cards = []
        self.matched_cards = []
        self.game_state = "ready_to_start"
        self.back_template = None
        
    def reset_game(self):
        """Reset game state"""
        self.cards = {}
        self.current_player = 1
        self.player_scores = {1: 0, 2: 0}
        self.turned_cards = []
        self.matched_cards = []
        self.game_state = "ready_to_start"
        self.back_template = None

    def get_turned_cards(self):
        return self.turned_cards
    
    def add_turned_card(self, card_id: int):
        if card_id not in self.turned_cards:
            self.turned_cards.append(card_id)
    
    def add_matched_cards(self, card1_id, card2_id):
        self.matched_cards.append((card1_id, card2_id))
        self.turned_cards.remove(card1_id)
        self.turned_cards.remove(card2_id)

    def remove_turned_card(self):
        self.turned_cards  = []


# Part 1. Data preparation

# calculate global resize scale
def calculate_global_scale(img_paths: list, target_pixels: int = 2_000_000) -> float:
    """
    calculate global resize scale based on the maximum number of pixels in the images
    
    Args:
        image_paths: list of image paths
        target_pixels: target number of pixels
    
    Returns:
        global resize scale
    """
    max_pixels = 0
    
    for img_path in img_paths:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                pixels = h * w
                max_pixels = max(max_pixels, pixels)
    
    if max_pixels > target_pixels:
        scale = (target_pixels / max_pixels) ** 0.5
        #print(f"Target pixels: {target_pixels}, Global resize scale: {scale:.3f}")
        return scale
    else:
        return 1.0

# Load and resize image
def load_and_preprocess_image(img_path: str, 
                              global_scale: float = 1.0) -> np.ndarray:
    """    
    Args:
        img_path: Path to the input image
        global_scale: Global resize scale (from calculate_global_scale)
    
    Returns:
        Preprocessed and enhanced image as numpy array

    Usage:
        image = load_and_preprocess_image(img_path, global_scale=scale)
    """
    # Check if file exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Try to load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # Get original dimensions
    original_h, original_w = image.shape[:2]
    
    # Apply uniform scaling
    if global_scale < 1.0:
        new_w = int(original_w * global_scale)
        new_h = int(original_h * global_scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        #print(f"Uniform resize: {original_w}×{original_h} -> {new_w}×{new_h} (scale: {global_scale:.3f})")

    # image enhancement
    try:
        # apply CLAHE to enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #print("Image enhancement applied")
    except Exception as e:
        print(f"Warning: Enhancement failed: {e}")
    
    return image

    
# Part 2. Card detection

def detect_cards(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect cards using contour detection method.
    
    Args:
        img: Input image (already scaled)
    
    Returns:
        List of card ids and bounding boxes as (id, (x, y, width, height))
    """
    # auto calculate area thresholds based on current image size
    img_area = img.shape[0] * img.shape[1]
    base_card_area = img_area / 30  # assume 30 cards can fit in the image
    min_area = int(base_card_area * 0.4) # unclosed contours are smaller
    max_area = int(base_card_area * 1.8) # angularly skewed cards are larger
    #min_area = int(base_card_area * 0.6)
    #max_area = int(base_card_area * 1.4)
    #print(f"Auto calculated area thresholds - min: {min_area}, max: {max_area}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # most robust # odd number
    #blurred = cv2.GaussianBlur(gray, (3, 3), 0) # 7 is the worst, then 3, 5 the best
    
    # Adaptive thresholding for better edge detection
    mean = np.mean(blurred)
    low_threshold = int(max(0, (1.0 - 0.35) * mean))
    high_threshold = int(min(255, (1.0 + 0.35) * mean))
    #low_threshold = int(max(0, (1.0 - 0.4) * mean))
    #high_threshold = int(min(255, (1.0 + 0.4) * mean))
    
    # Edge detection with adaptive thresholds
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # Morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    #kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1) # not necessary

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio
    raw_cards = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        if 0.5 < aspect_ratio < 2.0:
            raw_cards.append((x, y, w, h))

    print(f"Detected {len(raw_cards)} cards")

    cards = [{'id': i, 'bbox': box} for i, box in enumerate(raw_cards)] # id and bounding box for each card
    return cards

def extract_card_region(
        img: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        #margin_ratio: float = 0.1
) -> np.ndarray:
    """
    Extract card region from image based on bounding box
    
    Args:
        image: Input image
        bbox: (x, y, width, height) bounding box

    Returns:
        Card region
    """
    x, y, w, h = bbox
    #dx = int(w * margin_ratio)
    #dy = int(h * margin_ratio)
    #dx = 10
    #dy = 10

    # shrink box inwards by margin_ratio
    #x_new = max(0, x + dx)
    #y_new = max(0, y + dy)
    #w_new = max(1, w - 2*dx)
    #h_new = max(1, h - 2*dy)

    #return img[y_new:y_new+h_new, x_new:x_new+w_new]
    return img[y:y+h, x:x+w]


# Part 3. Card template creation

def create_back_template(img: np.ndarray, cards: List[Dict]) -> Optional[np.ndarray]:
    """
    Create template for card back by analyzing initial img
    
    Args:
        img: Input image
        cards: List of detected cards
        
    Returns:
        Template for card back detection
    """
    if len(cards) < 4:
        return None
    
    # Extract regions from cards (assumed to be face down)
    regions = []
    for card in cards:
        # Handle both dictionary and tuple formats
        if isinstance(card, dict):
            bbox = card['bbox']
        else:
            # If card is a tuple, assume format (x, y, w, h)
            bbox = card
        
        region = extract_card_region(img, bbox)
        if region.size > 0:
            # Resize to standard size
            region_resized = cv2.resize(region, (100, 150))
            regions.append(region_resized)
    
    if regions:
        # Average the regions to create template
        template = np.mean(regions, axis=0).astype(np.uint8)
        #return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        return template
    
    print("Failed to create back template")
    return None


# Part 4. Card state classification

def classify_card_state(
        card_region: np.ndarray, 
        back_template: Optional[np.ndarray] = None,
        template_threshold: float = 0.2, 
        color_variance_threshold: float = 5500,
        color_diff_threshold: float = 140
) -> str:
    """
    Classify if card is face up or face down
    
    Args:
        card_region: Cropped card image
        back_template: Template for card back (optional)
        
    Returns:
        "face_up" or "face_down"
    """
    if card_region.size == 0:
        return "unknown"

    gray_region = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
    # color variance analysis: Face-up cards typically have more color variance than uniform backs, but
    hsv_region = cv2.cvtColor(card_region, cv2.COLOR_BGR2HSV)
    color_variance = np.var(hsv_region[:,:,0]) + np.var(hsv_region[:,:,1]) + np.var(hsv_region[:,:,2])
    #print(f"Color variance: {color_variance}") # DEBUG
    color_diff = color_distance(hsv_region, back_template)
    #print(f"Color difference with back template: {color_diff}") # DEBUG
    #brightness = np.mean(cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY))
    #print(f"Brightness: {brightness}") # DEBUG

    # Template matching (if template available)
    max_corr = 0
    if back_template is not None:
        # Resize region to match template size
        resized_region = cv2.resize(gray_region, (back_template.shape[1], back_template.shape[0]))
        # convert to grayscale for template matching
        gray_template = cv2.cvtColor(back_template, cv2.COLOR_BGR2GRAY)
        # Compute normalized cross correlation
        correlation = cv2.matchTemplate(resized_region, gray_template, cv2.TM_CCOEFF_NORMED)
        max_corr = np.max(correlation)

        #print(f"Template matching correlation: {max_corr}") # DEBUG
        
    if max_corr > template_threshold:  # Threshold for back template match
        return "face_down"

    #if color_variance < color_threshold:
    if color_variance < color_variance_threshold:
        return "face_up"
    
    if color_diff > color_diff_threshold:
        return "face_up"
    else:
        return "face_down"
    

def color_distance(region1, region2) -> float:
    """
    Calculate color distance between two regions
    """
    hsv1 = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(region2, cv2.COLOR_BGR2HSV)

    mean1 = np.mean(hsv1.reshape(-1, 3), axis=0)
    mean2 = np.mean(hsv2.reshape(-1, 3), axis=0)

    diff = abs(mean1[0] - mean2[0])
    h_dist = min(diff, 180 - diff)
    s_dist = abs(mean1[1] - mean2[1])
    v_dist = abs(mean1[2] - mean2[2])

    #return np.sum(np.abs(mean1 - mean2))
    return h_dist * 2 +s_dist + v_dist *0.5

# Part 5. Card matching

def extract_sift_features(img: np.ndarray) -> Tuple[List, np.ndarray]:
    """
    Extract SIFT features from an image
    
    Args:
        img: Input image
    
    Returns:
        List of keypoints and descriptors
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect and compute SIFT features
    kp, des = sift.detectAndCompute(gray, None)
    
    return kp, des

def match_cards(
        card1_region: np.ndarray, 
        card2_region: np.ndarray, 
        threshold: int = 15
) -> bool:
    """
    Check if two cards match.
    
    Args:
        card1_region: Image region of first card
        card2_region: Image region of second card
        threshold: Matching confidence threshold
        
    Returns:
        True if cards match, False otherwise
    """
    # Detect and compute SIFT features
    kp1, des1 = extract_sift_features(card1_region)
    kp2, des2 = extract_sift_features(card2_region)
    #print(f"Feature count: {len(kp1)}, {len(kp2)}") # DEBUG

    if des1 is None or des2 is None:
        return False
    
    # use Lowe's ratio test
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    
    # apply Lowe's ratio test
    good_matches = []
    ratio_thresh = 0.75  # Lowe's ratio test threshold

    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    feature_count = np.mean([len(kp1), len(kp2)])
    threshold = int(feature_count * 0.3)
    threshold = max(threshold, 8)
    #if feature_count < 40:
    #    threshold = 10
    #else:
    #    threshold = 40

    #print(f"good matches: {len(good_matches)}") # DEBUG
    return len(good_matches) >= threshold

def draw_bounding_box(
        image: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        label: str,
        color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw a labeled bounding box on an image

    Args:
        image: Original image
        bbox: Bounding box (x, y, width, height)
        label: Text label to display
        color: Color tuple (B, G, R)
        
    Returns:
        Modified image with bounding box
    """
    result_img = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 5)
    cv2.putText(result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return result_img

def result_visualization(
    image: np.ndarray,
    card_list: List[Dict],
    match_pair: Optional[Tuple[int, int]] = None,
    match_result: Optional[bool] = None,
    game: Optional[MemoryGame] = None
) -> np.ndarray:
    """
    Visualize bounding box or match result on image
    
    Args:
        image: Original image
        card_list: Detected cards [{'id': ..., 'bbox': (...)}]
        match_pair: Matching card id (id1, id2), optional
        match_result: True for match, False for no match, optional
        game: MemoryGame instance for addtional state info, optional

    Returns:
        Image with highlighted result
    """
    result_img = image.copy()

    # if match pair and results are provided
    if match_pair is not None and match_result is not None:
        id1, id2 = match_pair
        color = (30, 200, 30) if match_result else (30, 30, 200)  # green / red

        for idx in (id1, id2):
            card = card_list[idx]
            x, y, w, h = card['bbox']
            label = f"Card {card['id']}"
            result_img = draw_bounding_box(result_img, (x, y, w, h), label, color)

    # if no match pair and results are provided
    else:
        for card in card_list:
            x, y, w, h = card['bbox']
            label = f"Card {card['id']}"
            color = (255, 70, 30) #(255, 130, 85) # blue
            result_img = draw_bounding_box(result_img, (x, y, w, h), label, color)

    return result_img


# Part 6. Game state management
def update_game_state(
        game: MemoryGame, 
        image: np.ndarray, 
        cards: List[Dict]
) -> Tuple[str, Optional[Tuple[int, int]], Optional[bool]]:
    """
    Update game state based on current image
    
    Args:
        game: MemoryGame instance
        image: Current image
        cards: List of card bounding boxes
        
    Returns:
        Tuple containing:
        - Game instruction message
        - Match pair (id1, id2) if two cards are turned, None otherwise
        - Match result if two cards are turned, None otherwise
    """
    message = ""
    match_pair = None
    match_result = None

    # game over check
    if len(cards) == 0:
        total_pairs = game.player_scores[1] + game.player_scores[2]
        if total_pairs == 8:
            if game.player_scores[1] == game.player_scores[2]:
                return f"\nPlayer 1 score: {game.player_scores[1]} pairs \nPlayer 2 score: {game.player_scores[2]} pairs \nIt's a draw!"
            else:
                winner = 1 if game.player_scores[1] > game.player_scores[2] else 2
                return f"\nPlayer 1 score: {game.player_scores[1]} pairs \nPlayer 2 score: {game.player_scores[2]} pairs \nWinner is Player {winner}."

    # game logic
    if len(game.turned_cards) == 0:
        message = f"Player {game.current_player}, please choose two cards to turn."
    elif len(game.turned_cards) == 2:
        print(f"Face up card ids: {game.turned_cards}")
        id1, id2 = game.turned_cards
        card1_region = game.cards[game.turned_cards[0]]['region']
        card2_region = game.cards[game.turned_cards[1]]['region']
        
        match_pair = (id1, id2)
        match_result = match_cards(card1_region, card2_region)

        if match_result:
            game.add_matched_cards(id1, id2)
            game.player_scores[game.current_player] += 1
            message = "Congratulations! A matching pair. Please pick up your matching card pair and then you may continue."
        else:
            # switch player
            game.current_player = 2 if game.current_player == 1 else 1
            message = f"No match! Turn back the two cards face down again. Player {game.current_player} may then continue."
    else:
        message = f"Player {game.current_player}, please choose two cards to turn."
    
    return message, match_pair, match_result


# Part 7. Performance evaluation
#def detection_accuracy(detected_cards: List[Tuple], ground_truth_cards: List[Dict]) -> float:
def detection_count_accuracy(detected_cards: int, ground_truth_count: int) -> float:
    """
    Calculate detection count accuracy
    """
    #if len(detected_cards) != len(ground_truth_cards):
    #    return 0.0
    
    #correct = 0
    #for detected_card, ground_truth_card in zip(detected_cards, ground_truth_cards):
   #     if detected_card['id'] == ground_truth_card['id']:
   #         correct += 1
    
    #return correct / len(detected_cards)
    return 1.0 * 100 if len(detected_cards) == ground_truth_count else 0.0

def classification_accuracy(game: MemoryGame, ground_truth_state: Dict[int, str]) -> float:
    """
    Calculate classification accuracy
    """    
    correct = 0
    total = 0

    for card_id, state in ground_truth_state.items():
        if game.cards[card_id]['state'] == state:
            correct += 1
        total += 1

    return (correct / total) * 100 if total > 0 else 0.0

def match_accuracy(match_attempts: List[Dict]) -> float:
    """
    Calculate match accuracy
    """
    if len(match_attempts) == 0:
        return 0.0
    
    correct = sum(1 for attempt in match_attempts
                  if attempt['prediction'] == attempt['ground_truth'])
    
    return (correct / len(match_attempts)) * 100 if len(match_attempts) > 0 else 0.0

