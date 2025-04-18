import os
import requests
import base64
from urllib.parse import urljoin


def fetch_github_repo(owner, repo, github_token=None):
    """
    Fetch a GitHub repository content and save it to a text file with original structure.

    Args:
        owner (str): GitHub repository owner/username
        repo (str): Repository name
        github_token (str, optional): GitHub personal access token for authentication
    """
    # API base URL
    base_url = "https://api.github.com/"
    api_url = urljoin(base_url, f"repos/{owner}/{repo}/contents")

    # Headers for authentication if token is provided
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    # Output file
    output_file = f"{repo}_structure.txt"

    # Track all directories for folder structure visualization
    all_directories = set()
    all_files = set()

    def fetch_content(path="", indent_level=0):
        """Recursively fetch repository content"""
        url = api_url
        if path:
            url = f"{api_url}/{path}"

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Error accessing {url}: {response.status_code}")
            print(response.json().get("message", ""))
            return

        contents = response.json()

        # Handle case when API returns a single file instead of a list
        if not isinstance(contents, list):
            contents = [contents]

        # Sort contents: directories first, then files
        directories = [item for item in contents if item["type"] == "dir"]
        files = [item for item in contents if item["type"] == "file"]

        # Process directories first
        for item in sorted(directories, key=lambda x: x["path"]):
            item_path = item["path"]

            # Skip .git directories
            if ".git" in item_path:
                continue

            # Add directory to our structure tracking
            all_directories.add(item_path)

            # Recursively fetch directory contents
            fetch_content(item_path, indent_level + 1)

        # Then process files
        for item in sorted(files, key=lambda x: x["path"]):
            item_path = item["path"]

            # Skip .git files
            if ".git" in item_path:
                continue

            # Add file to our tracking
            all_files.add(item_path)

            # Download and save file content
            try:
                if item.get("download_url"):
                    file_response = requests.get(item["download_url"])
                    content = file_response.text
                else:
                    # For larger files, content is provided as base64
                    content = base64.b64decode(item["content"]).decode("utf-8")

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n\n### FILE: {item_path} ###\n")
                    f.write(content)
                    f.write("\n")
            except Exception as e:
                print(f"Error processing file {item_path}: {str(e)}")

    # Create or clear the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Repository: {owner}/{repo}\n")

    # Start fetching from root
    fetch_content()

    # After fetching all content, write the folder structure at the beginning
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Repository: {owner}/{repo}\n\n")
        f.write("## FOLDER STRUCTURE\n")

        # Write root directory
        f.write(".\n")

        # Create a simpler folder structure visualization
        dir_list = sorted(list(all_directories))

        # Process each directory to create visual structure
        for directory in dir_list:
            # Split path into components
            parts = directory.split('/')
            depth = len(parts)

            # Simple indentation based on depth
            indent = "    " * (depth - 1)
            f.write(f"{indent}├── {parts[-1]}/\n")

        # Now list all files with proper indentation
        f.write("\n## FILES\n")
        for file_path in sorted(list(all_files)):
            # Split path into components
            parts = file_path.split('/')
            filename = parts[-1]
            depth = len(parts)

            # Simple indentation based on depth
            indent = "    " * (depth - 1)
            f.write(f"{indent}├── {filename}\n")

        f.write("\n## FILE CONTENTS\n")
        f.write(content[content.find('#'):])  # Skip the first line which we've already written

    print(f"Repository structure written to {output_file}")


if __name__ == "__main__":
    # User inputs
    owner = input("Enter GitHub repository owner/username: ")
    repo = input("Enter repository name: ")
    use_token = input("Do you have a GitHub token? (y/n): ").lower() == 'y'

    token = None
    if use_token:
        token = input("Enter your GitHub token (or leave blank): ")
        if not token:
            token = None

    fetch_github_repo(owner, repo, token)