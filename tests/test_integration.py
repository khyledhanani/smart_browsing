import os
import unittest
from agent import WebAgent

class TestWebAgentIntegration(unittest.TestCase):
    def setUp(self):
        self.agent = WebAgent()
        
    def tearDown(self):
        self.agent.close()
        
    def test_navigate_and_capture(self):
        # Test navigation and screenshot capture
        try:
            self.agent.navigate("https://www.python.org")
        except Exception as e:
            self.fail(f"Navigation failed with: {e}")
        
        screenshot = self.agent.capture_screenshot()
        self.assertTrue(os.path.exists(screenshot), "Screenshot file does not exist")
        
        # Cleanup screenshot
        if os.path.exists(screenshot):
            os.remove(screenshot)
            
    def test_analyze_image(self):
        # Test analyze_image returns expected result structure
        self.agent.navigate("https://www.python.org")
        screenshot = self.agent.capture_screenshot()
        analysis = self.agent.analyze_image(screenshot)
        
        self.assertIn('gpt_analysis', analysis)
        self.assertIn('detected_elements', analysis)
        detected = analysis['detected_elements']
        self.assertIn('buttons', detected)
        self.assertIn('text_areas', detected)
        self.assertIn('clickable', detected)
        
        # Cleanup screenshot
        if os.path.exists(screenshot):
            os.remove(screenshot)

if __name__ == "__main__":
    unittest.main() 