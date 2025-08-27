import pyautogui
import subprocess
import time
import unittest
import os

class TestFaceRecognitionApp(unittest.TestCase):

    def setUp(self):
        self.exe_path = r"dist\FaceRecognitionApp\FaceRecognitionApp.exe"
        self.process = subprocess.Popen(self.exe_path)
        time.sleep(5)  # Wait for GUI to load

    def test_invalid_login(self):
        """Test invalid login using screen positions"""
        # Replace with your actual screen coordinates
        username_pos = (400, 300)
        password_pos = (400, 360)
        login_btn_pos = (400, 440)

        pyautogui.click(username_pos)
        pyautogui.write("invaliduser")

        pyautogui.click(password_pos)
        pyautogui.write("wrongpass")

        pyautogui.click(login_btn_pos)
        time.sleep(2)

    def test_valid_login(self):
        """Test valid login"""
        username_pos = (400, 300)
        password_pos = (400, 360)
        login_btn_pos = (400, 440)

        pyautogui.click(username_pos)
        pyautogui.write("testadmin")

        pyautogui.click(password_pos)
        pyautogui.write("securepass")

        pyautogui.click(login_btn_pos)
        time.sleep(4)

        # Optionally take a screenshot for verification
        screenshot = pyautogui.screenshot()
        screenshot.save("screenshot_after_login.png")

    def tearDown(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

if __name__ == "__main__":
    unittest.main()
