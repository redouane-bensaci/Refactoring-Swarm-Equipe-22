import os
from typing import Optional


class FileHandlerService():
    def read_file_to_text(self, file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Reads the content of a given file path safely and returns it as a string.

        This function includes robust error handling for common file I/O issues.

        Args:
            file_path (str): The path to the file you want to read.
            encoding (str): The character encoding of the file (default 'utf-8').

        Returns:
            Optional[str]: The entire content of the file as a text string if successful, 
                        otherwise returns None and prints the error message.
        """
        # Basic input validation
        if not isinstance(file_path, str) or not file_path:
            print("Error: Invalid file path provided.")
            return None

        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            return None
        
        try:
            # Use 'with open()' for automatic resource management
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        
        except FileNotFoundError:
            # This is caught by os.path.exists() check above, but kept for absolute safety
            print(f"Error: File not found at '{file_path}'.")
            return None
        except IOError as e:
            # Catches permission errors, disk full errors, etc.
            print(f"IO Error reading file '{file_path}': {e}")
            return None
        except Exception as e:
            # Catches all other unforeseen errors
            print(f"An unexpected error occurred while reading the file: {e}")
            return None


    def write_text_to_file(self, file_path: str, text_content: str, mode: str = 'w', encoding: str = 'utf-8') -> bool:
        """
        Writes text content to a specified file path.

        Args:
            file_path (str): The path to the destination file.
            text_content (str): The text string to write into the file.
            mode (str): File open mode. 
                        'w' = write mode (overwrites existing file or creates new) (default).
                        'a' = append mode (adds to the end of the file).
                        'x' = exclusive creation mode (fails if file already exists).
            encoding (str): The character encoding to use (default 'utf-8').

        Returns:
            bool: True if the write operation was successful, False otherwise.
        """
        # Basic input validation
        if not isinstance(file_path, str) or not file_path:
            print("Error: Invalid file path provided.")
            return False
        
        if not isinstance(text_content, str):
            print("Error: Content to write must be a string.")
            return False
            
        try:
            # Ensure the directory exists before trying to open the file
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Created missing directory: {directory}")

            with open(file_path, mode, encoding=encoding) as file:
                file.write(text_content)
            
            # print(f"Successfully wrote content to '{file_path}' using mode '{mode}'.")
            return True

        except FileExistsError:
            print(f"Error: File '{file_path}' already exists (when using mode='x').")
            return False
        except IOError as e:
            print(f"IO Error writing file '{file_path}': {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while writing the file: {e}")
            return False



#instantiation
file_service = FileHandlerService()