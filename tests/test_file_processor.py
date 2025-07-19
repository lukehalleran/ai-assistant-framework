# daemon_7_11_25_refactor/test_file_processor.py
import asyncio
import tempfile
import os
from utils.file_processor import FileProcessor

# Create a mock file object
class MockFile:
    def __init__(self, name, content):
        self.name = name
        # Write content to a temp file
        with open(name, 'w') as f:
            f.write(content)

async def test_file_processor():
    processor = FileProcessor()

    # Create test files
    test_files = []

    # Create a text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content from a text file.")
        test_files.append(MockFile(f.name, ""))

    # Create a Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def hello():\n    print('Hello from Python!')")
        test_files.append(MockFile(f.name, ""))

    # Test processing
    user_text = "Please analyze these files:"
    result = await processor.process_files(user_text, test_files)

    print("Processed result:")
    print("-" * 50)
    print(result)
    print("-" * 50)

    # Cleanup
    for file in test_files:
        os.unlink(file.name)

    print("\n✅ File processor test complete!")

if __name__ == "__main__":
    asyncio.run(test_file_processor())
