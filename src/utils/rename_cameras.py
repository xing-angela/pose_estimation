import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Rename Cameras')
    parser.add_argument("--root_dir", type=str, default="./dataset", help="Root directory of the dataset")
    parser.add_argument("--seq", type=str, default="maracas", help="Sequence path")
    parser.add_argument("--target_dir", type=str, default="image", help="Target directory")
    parser.add_argument("--face_to_cam_path", type=str, default="./assets/face_to_cam.json", help="Path to the face to camera mapping")
    args = parser.parse_args()

    with open(args.face_to_cam_path) as file:
        face_to_cam = json.load(file)
    
    images_path = os.path.join(args.root_dir, args.seq, "data", args.target_dir)
    for face, cam in face_to_cam.items():
        orig = os.path.join(images_path, cam)
        if os.path.exists(orig):
            print(f"Renaming {cam} to {face}")
            new = os.path.join(images_path, face)
            os.rename(orig, new)

if __name__ == '__main__':
    main()