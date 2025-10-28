import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import gradio as gr
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ChronoLens:
    """Visual Difference Engine for detecting and classifying changes between time-series images."""
    
    def __init__(self):
        self.min_match_count = 10
        self.change_threshold = 0.02  # 2% of image area
        
    def align_images(self, image_a, image_b):
        """
        Robust alignment of image_b to image_a using feature-based matching with fallback.
        
        Args:
            image_a: Reference image (numpy array)
            image_b: Image to align (numpy array)
            
        Returns:
            aligned_image_b: Aligned version of image_b
            success: Boolean indicating if alignment was successful
        """
        # Ensure images are the same size initially
        if image_a.shape[:2] != image_b.shape[:2]:
            image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]))
        
        # Convert to grayscale for feature detection
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY) if len(image_a.shape) == 3 else image_a
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY) if len(image_b.shape) == 3 else image_b
        
        # Try feature-based alignment first (ORB + RANSAC)
        try:
            aligned, success = self._feature_based_alignment(gray_a, gray_b, image_b)
            if success:
                return aligned, True
        except Exception as e:
            print(f"Feature-based alignment failed: {e}")
        
        # Fallback to ECC (Enhanced Correlation Coefficient)
        try:
            aligned, success = self._ecc_alignment(gray_a, gray_b, image_b)
            if success:
                return aligned, True
        except Exception as e:
            print(f"ECC alignment failed: {e}")
        
        # Final fallback: return resized image_b
        print("All alignment methods failed. Using direct resize.")
        return image_b, False
    
    def _feature_based_alignment(self, gray_a, gray_b, image_b):
        """ORB feature matching with RANSAC homography estimation."""
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=5000)
        
        # Detect keypoints and descriptors
        kp_a, desc_a = orb.detectAndCompute(gray_a, None)
        kp_b, desc_b = orb.detectAndCompute(gray_b, None)
        
        if desc_a is None or desc_b is None:
            return image_b, False
        
        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc_b, desc_a, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_match_count:
            return image_b, False
        
        # Extract location of good matches
        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
        
        if H is None:
            return image_b, False
        
        # Warp image_b to align with image_a
        h, w = gray_a.shape
        aligned = cv2.warpPerspective(image_b, H, (w, h))
        
        return aligned, True
    
    def _ecc_alignment(self, gray_a, gray_b, image_b):
        """ECC-based alignment as fallback."""
        # Define motion model (EUCLIDEAN for translation + rotation)
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        
        # Run ECC algorithm
        try:
            _, warp_matrix = cv2.findTransformECC(gray_a, gray_b, warp_matrix, warp_mode, criteria)
            
            # Warp image_b
            h, w = gray_a.shape
            aligned = cv2.warpAffine(image_b, warp_matrix, (w, h), 
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            return aligned, True
        except:
            return image_b, False
    
    def detect_changes(self, image_a, image_b_aligned):
        """
        Detect changes using pixel difference and SSIM.
        
        Args:
            image_a: Reference image
            image_b_aligned: Aligned current image
            
        Returns:
            change_mask: Binary mask of detected changes
            combined_score: Overall change score
        """
        # Convert to grayscale
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY) if len(image_a.shape) == 3 else image_a
        gray_b = cv2.cvtColor(image_b_aligned, cv2.COLOR_BGR2GRAY) if len(image_b_aligned.shape) == 3 else image_b_aligned
        
        # Ensure same size
        if gray_a.shape != gray_b.shape:
            gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
        
        # 1. Pixel Difference
        diff = cv2.absdiff(gray_a, gray_b)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 2. SSIM (Structural Similarity Index)
        ssim_score, ssim_img = ssim(gray_a, gray_b, full=True)
        ssim_img = (1 - ssim_img) * 255  # Convert to dissimilarity
        ssim_img = ssim_img.astype(np.uint8)
        _, ssim_mask = cv2.threshold(ssim_img, 50, 255, cv2.THRESH_BINARY)
        
        # 3. Combined mask (weighted fusion)
        combined = cv2.addWeighted(diff_mask, 0.5, ssim_mask, 0.5, 0)
        _, combined_mask = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
        
        # 4. Morphological refinement
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate overall change score
        change_pixels = np.count_nonzero(combined_mask)
        total_pixels = combined_mask.size
        change_score = change_pixels / total_pixels
        
        return combined_mask, change_score
    
    def classify_changes(self, change_score):
        """
        Simple classification based on change score.
        
        Args:
            change_score: Proportion of changed pixels
            
        Returns:
            classification: String label
            confidence: Confidence percentage
        """
        if change_score < self.change_threshold:
            return "No Significant Change", (1 - change_score / self.change_threshold) * 100
        elif change_score < 0.1:
            return "Minor Change Detected", change_score * 1000
        elif change_score < 0.3:
            return "Moderate Change Detected", min(change_score * 300, 100)
        else:
            return "Major Change / Potential Anomaly", min(change_score * 200, 100)
    
    def create_overlay(self, image, mask, color=(0, 0, 255), alpha=0.4):
        """
        Create a visual overlay of the change mask on the image.
        
        Args:
            image: Base image
            mask: Binary mask
            color: RGB color for overlay (default: red)
            alpha: Transparency factor
            
        Returns:
            overlay_image: Image with overlay
        """
        overlay = image.copy()
        
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Create colored overlay
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        return overlay
    
    def process(self, image_a_path, image_b_path):
        """
        Main processing pipeline for ChronoLens.
        
        Args:
            image_a_path: Path to reference image
            image_b_path: Path to current image
            
        Returns:
            Tuple of output images and classification text
        """
        try:
            # Load images
            image_a = cv2.imread(image_a_path)
            image_b = cv2.imread(image_b_path)
            
            if image_a is None or image_b is None:
                return None, None, None, "Error: Could not load images"
            
            # Convert BGR to RGB for display
            image_a_rgb = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
            image_b_rgb = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
            
            # Step 1: Align images
            aligned_b, alignment_success = self.align_images(image_a, image_b)
            aligned_b_rgb = cv2.cvtColor(aligned_b, cv2.COLOR_BGR2RGB)
            
            # Step 2: Detect changes
            change_mask, change_score = self.detect_changes(image_a, aligned_b)
            
            # Step 3: Classify changes
            classification, confidence = self.classify_changes(change_score)
            
            # Step 4: Create visualization
            overlay_rgb = self.create_overlay(aligned_b_rgb, change_mask)
            
            # Create result text
            alignment_status = "✓ Successful" if alignment_success else "⚠ Fallback Mode"
            result_text = f"""
            **ChronoLens Analysis Results**
            
            **Alignment Status:** {alignment_status}
            **Change Detection:** {change_score*100:.2f}% of image modified
            **Classification:** {classification}
            **Confidence:** {confidence:.1f}%
            
            ---
            *Analysis complete. Red overlay indicates detected changes.*
            """
            
            return image_a_rgb, aligned_b_rgb, overlay_rgb, result_text
            
        except Exception as e:
            return None, None, None, f"Error during processing: {str(e)}"


def create_gradio_interface():
    """Create Gradio interface for ChronoLens."""
    
    engine = ChronoLens()
    
    def process_wrapper(image_a, image_b):
        """Wrapper function for Gradio interface."""
        if image_a is None or image_b is None:
            return None, None, None, "Please upload both images."
        
        return engine.process(image_a, image_b)
    
    # Create interface
    interface = gr.Interface(
        fn=process_wrapper,
        inputs=[
            gr.Image(type="filepath", label="Reference Image (Before)"),
            gr.Image(type="filepath", label="Current Image (After)")
        ],
        outputs=[
            gr.Image(label="Reference Image"),
            gr.Image(label="Aligned Current Image"),
            gr.Image(label="Change Detection Overlay"),
            gr.Markdown(label="Analysis Results")
        ],
        title="🔍 IgnitionX - Visual Difference Engine",
        description="""
        **Detect and classify visual changes between time-series images.**
        
        Upload a 'before' (reference) and 'after' (current) image to analyze changes.
        ChronoLens will automatically align the images and highlight detected differences.
        
        *Features: Robust alignment • Multi-method change detection • Smart classification*
        """,
        examples=[],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface


if __name__ == "__main__":
    # Launch the interface
    interface = create_gradio_interface()
    interface.launch(share=True)