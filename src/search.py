import os
import argparse


def print_files_and_folders(path):
    for root, directories, files in os.walk(path):
        # Print the current directory
        print(f"Directory: {root}")

        # Print the subdirectories in the current directory
        print("Subdirectories:")
        for directory in directories:
            print(directory)

        print("")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)

    args = parser.parse_args()
    directory_path = args.path

    print_files_and_folders(directory_path)