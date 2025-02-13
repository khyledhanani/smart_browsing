from agent import WebAgent
import cv2

def main():
    agent = WebAgent()
    try:
        # Example task
        agent.navigate("https://www.python.org")  # Using Python's website as an example
        screenshot = agent.capture_screenshot()
        analysis = agent.analyze_image(screenshot)
        
        # Print analysis results
        print("\nGPT-4 Analysis:")
        print(analysis['gpt_analysis'])
        
        print("\nDetected Elements:")
        print(f"Found {len(analysis['detected_elements']['buttons'])} buttons")
        print(f"Found {len(analysis['detected_elements']['text_areas'])} text areas")
        print(f"Found {len(analysis['detected_elements']['clickable'])} clickable elements")
        
        # Visualize detections on the image
        image = cv2.imread(screenshot)
        for button in analysis['detected_elements']['buttons']:
            x, y, w, h = button['position']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        for text_area in analysis['detected_elements']['text_areas']:
            x, y, w, h = text_area['position']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        cv2.imwrite('analyzed_screenshot.png', image)
        
    finally:
        agent.close()

if __name__ == "__main__":
    main() 