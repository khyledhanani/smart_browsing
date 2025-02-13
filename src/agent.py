from playwright.sync_api import sync_playwright
import cv2
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
from typing import List, Tuple
import pytesseract  # We'll need this for OCR

class WebAgent:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.page = self.browser.new_page()
    
    def capture_screenshot(self):
        """Capture screenshot and save it temporarily"""
        self.page.screenshot(path="temp_screenshot.png")
        return "temp_screenshot.png"
    
    def analyze_image(self, image_path):
        """Analyze image using OpenCV and GPT-4 Vision"""
        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        # Detect various elements
        buttons = self._detect_buttons(image)
        text_areas = self._detect_text_areas(image)
        clickable = self._detect_clickable_elements(image)
        
        # Combine analysis results
        analysis_results = {
            'buttons': buttons,
            'text_areas': text_areas,
            'clickable': clickable
        }
        
        # Send enhanced information to GPT-4 Vision
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
            
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this webpage. I've detected the following elements:
                            Buttons: {len(buttons)}
                            Text Areas: {len(text_areas)}
                            Clickable Elements: {len(clickable)}
                            
                            Please describe what you see and how these elements relate to the page content."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        )
        return {
            'gpt_analysis': response.choices[0].message.content,
            'detected_elements': analysis_results
        }

    def _detect_buttons(self, image: np.ndarray) -> List[dict]:
        """Detect button-like elements in the image"""
        buttons = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            
            # Buttons typically have specific aspect ratios and sizes
            if 1.5 <= aspect_ratio <= 5 and w >= 50 and h >= 20:
                roi = image[y:y+h, x:x+w]
                # Extract text from the button using OCR
                text = pytesseract.image_to_string(roi).strip()
                
                buttons.append({
                    'position': (x, y, w, h),
                    'text': text,
                    'confidence': self._calculate_button_confidence(roi)
                })
        
        return buttons

    def _detect_text_areas(self, image: np.ndarray) -> List[dict]:
        """Detect text areas in the image"""
        text_areas = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours of text blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        binary = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 10:  # Minimum size threshold
                roi = image[y:y+h, x:x+w]
                text = pytesseract.image_to_string(roi).strip()
                
                if text:  # Only add if text was detected
                    text_areas.append({
                        'position': (x, y, w, h),
                        'text': text
                    })
        
        return text_areas

    def _detect_clickable_elements(self, image: np.ndarray) -> List[dict]:
        """Detect potentially clickable elements (links, buttons, icons)"""
        clickable = []
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common link colors
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Create mask for blue colors (common for links)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Find contours of blue elements
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 10:  # Minimum size threshold
                roi = image[y:y+h, x:x+w]
                text = pytesseract.image_to_string(roi).strip()
                
                clickable.append({
                    'position': (x, y, w, h),
                    'text': text,
                    'type': 'link'
                })
        
        return clickable

    def _calculate_button_confidence(self, roi: np.ndarray) -> float:
        """Calculate confidence score for button detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate various features
        aspect_ratio = float(roi.shape[1])/roi.shape[0]
        has_border = self._detect_border(gray)
        has_text = bool(pytesseract.image_to_string(roi).strip())
        
        # Simple scoring system
        score = 0.0
        if 2.0 <= aspect_ratio <= 4.0: score += 0.3
        if has_border: score += 0.3
        if has_text: score += 0.4
        
        return score

    def _detect_border(self, gray_image: np.ndarray) -> bool:
        """Detect if an element has a border"""
        edges = cv2.Canny(gray_image, 50, 150)
        return np.sum(edges) > 0
    
    def navigate(self, url):
        """Navigate to a specific URL"""
        self.page.goto(url)
    
    def close(self):
        """Close browser and cleanup"""
        self.browser.close()
        self.playwright.stop()

    def execute_task(self, task_description):
        """Execute a specific task based on description by planning multi-step actions."""
        print("Executing Task:", task_description)
        # Capture a screenshot of the current page
        screenshot = self.capture_screenshot()

        # Perform analysis of the screenshot (including button, text area, and clickable element detection)
        analysis = self.analyze_image(screenshot)

        # Build a prompt combining the task description with the analysis results
        prompt = f"""
        You are an intelligent AI agent that can interact with a webpage through browser automation.
        
        Task: {task_description}
        
        Current page analysis:
        GPT-4 Visual Analysis: {analysis['gpt_analysis']}
        Detected Elements: 
        - Buttons: {len(analysis['detected_elements']['buttons'])} detected.
        - Text Areas: {len(analysis['detected_elements']['text_areas'])} detected.
        - Clickable Elements: {len(analysis['detected_elements']['clickable'])} detected.
        
        Based on this information, please provide a detailed multi-step plan to accomplish the task.
        For each step, specify the action (e.g., click a button, navigate to a URL, fill a form) and the reasoning behind it.
        """

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        plan = response.choices[0].message.content
        print("Proposed Plan:\n", plan)
        return plan

    # New Playwright functionalities:

    def click_element(self, selector: str, timeout: int = 5000):
        """Click the element on the page identified by the CSS selector."""
        self.page.wait_for_selector(selector, timeout=timeout)
        self.page.click(selector)

    def fill_form_field(self, selector: str, value: str, timeout: int = 5000):
        """Fill in a form field identified by the CSS selector with the specified value."""
        self.page.wait_for_selector(selector, timeout=timeout)
        self.page.fill(selector, value)

    def wait_for_dynamic_content(self, selector: str, timeout: int = 5000):
        """Wait for dynamic content to be loaded based on the CSS selector."""
        return self.page.wait_for_selector(selector, timeout=timeout)

    def handle_popup(self):
        """Handle a popup window using Playwright's event listener."""
        with self.page.expect_popup() as popup_info:
            # This block should enclose the action that triggers the popup.
            pass
        popup = popup_info.value
        print("Popup detected:", popup.url)
        # Do any required interactions with the popup here.
        popup.close()
        return popup

    def switch_to_frame(self, frame_name: str):
        """Switch context to an iframe within the current page by its name attribute."""
        frame = self.page.frame(name=frame_name)
        if frame is None:
            print("No frame found with name:", frame_name)
        else:
            print("Switched to frame:", frame_name)
        return frame 